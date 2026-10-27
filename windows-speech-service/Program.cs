using System.Text.Json;
using Microsoft.Windows.AI;
using Microsoft.Windows.AI.Speech;

// Isolated from the WinUI host and Python's ONNX Runtime. No microphone, network
// recognizer, or recording is created here. Audio is supplied by Sunno's capture.
internal static class Program
{
    private static SpeechRecognitionModel? model;
    private static StreamingRecognition? recognizer;
    private static SpeechAudioProvider? provider;
    private static readonly object textLock = new();
    private static readonly SortedDictionary<long, string> completed = new();
    private static string provisional = "";
    private static long generation;

    private static async Task Main()
    {
        string? line;
        while ((line = await Console.In.ReadLineAsync()) != null)
        {
            try
            {
                if (line.Length > 4 * 1024 * 1024)
                    throw new InvalidOperationException("Speech request exceeds the audio limit");
                using var message = JsonDocument.Parse(line);
                var root = message.RootElement;
                var command = root.GetProperty("command").GetString();
                object response;
                switch (command)
                {
                    case "probe":
                    {
                        var state = SpeechRecognitionModelFactory.Default.GetReadyState();
                        bool supported = state is AIFeatureReadyState.Ready or AIFeatureReadyState.NotReady;
                        response = new { ok = true, supported,
                            available = state == AIFeatureReadyState.Ready,
                            ready_state = state.ToString(),
                            reason = ReadinessMessages.ForState(state.ToString()) };
                        break;
                    }
                    case "prepare":
                    {
                        var state = SpeechRecognitionModelFactory.Default.GetReadyState();
                        if (state == AIFeatureReadyState.Ready)
                        {
                            response = new { ok = true };
                            break;
                        }
                        if (state != AIFeatureReadyState.NotReady)
                            throw new InvalidOperationException(ReadinessMessages.ForState(state.ToString()));
                        var result = await SpeechRecognitionModelFactory.Default.EnsureReadyAsync();
                        state = SpeechRecognitionModelFactory.Default.GetReadyState();
                        if (state != AIFeatureReadyState.Ready)
                        {
                            // EnsureReadyAsync can finish while Windows Update is still
                            // installing the CPU model. Do not call that a failed download.
                            if (result.Status != AIFeatureReadyResultState.Failure)
                            {
                                var deadline = DateTime.UtcNow.AddMinutes(3);
                                while (state == AIFeatureReadyState.NotReady && DateTime.UtcNow < deadline)
                                {
                                    await Task.Delay(TimeSpan.FromSeconds(2));
                                    state = SpeechRecognitionModelFactory.Default.GetReadyState();
                                }
                            }
                            if (state != AIFeatureReadyState.Ready)
                            {
                                response = new { ok = false,
                                    error = "Windows AI speech: " + ReadinessMessages.ForSetup(
                                        state.ToString(), result.Status.ToString(), result.ErrorDisplayText,
                                        FormatError(result.Error), FormatError(result.ExtendedError),
                                        result.PackageInstallationFailed),
                                    ready_state = state.ToString(), setup_status = result.Status.ToString(),
                                    setup_error = FormatError(result.Error),
                                    extended_error = FormatError(result.ExtendedError),
                                    package_installation_failed = result.PackageInstallationFailed };
                                break;
                            }
                        }
                        response = new { ok = true };
                        break;
                    }
                    case "load":
                        var readyState = SpeechRecognitionModelFactory.Default.GetReadyState();
                        if (readyState != AIFeatureReadyState.Ready)
                            throw new InvalidOperationException(ReadinessMessages.ForState(readyState.ToString()));
                        model = await SpeechRecognitionModelFactory.Default.CreateAsync();
                        response = new { ok = true };
                        break;
                    case "start":
                        await Reset();
                        if (model == null) throw new InvalidOperationException("Speech model is not loaded");
                        provider = new SpeechAudioProvider();
                        recognizer = new StreamingRecognition(AudioConfiguration.ForProvider(provider), model);
                        var streamGeneration = generation;
                        recognizer.Recognizing += (_, e) =>
                        {
                            lock (textLock)
                                if (streamGeneration == generation) provisional = e.Text;
                        };
                        recognizer.Recognized += (_, e) =>
                        {
                            lock (textLock)
                            {
                                if (streamGeneration != generation) return;
                                if (e.IsFinal)
                                {
                                    completed[(long)(e.Offset * 1000000.0)] = e.Text;
                                    provisional = "";
                                }
                                else provisional = e.Text;
                            }
                        };
                        await recognizer.StartContinuousRecognitionAsync();
                        response = new { ok = true };
                        break;
                    case "audio":
                        if (provider == null || recognizer == null)
                            throw new InvalidOperationException("No active speech stream");
                        var bytes = Convert.FromBase64String(root.GetProperty("samples").GetString() ?? "");
                        if ((bytes.Length & 1) != 0 || bytes.Length > 16000 * 2 * 30)
                            throw new InvalidOperationException("Invalid 16 kHz PCM audio");
                        var pcm = new ushort[bytes.Length / 2];
                        Buffer.BlockCopy(bytes, 0, pcm, 0, bytes.Length);
                        if (pcm.Length > 0) provider.PushData(pcm);
                        bool final = root.GetProperty("final").GetBoolean();
                        if (final)
                        {
                            // StopContinuousRecognitionAsync flushes pending recognition.
                            await recognizer.StopContinuousRecognitionAsync().AsTask()
                                .WaitAsync(TimeSpan.FromSeconds(20));
                        }
                        string text;
                        lock (textLock)
                            text = string.Join(" ", completed.Values.Append(provisional)
                                .Where(value => !string.IsNullOrWhiteSpace(value))).Trim();
                        if (final) await Reset();
                        response = new { ok = true, text };
                        break;
                    case "reset":
                        await Reset();
                        response = new { ok = true };
                        break;
                    default:
                        throw new InvalidOperationException("Unknown speech command");
                }
                Console.WriteLine(JsonSerializer.Serialize(response));
            }
            catch (Exception exception)
            {
                Console.WriteLine(JsonSerializer.Serialize(new { ok = false,
                    error = $"Windows AI speech: {exception.Message}" }));
            }
        }
        await Reset();
        model?.Dispose();
    }

    private static string FormatError(object? value) => value switch
    {
        null => "",
        Exception exception => string.IsNullOrWhiteSpace(exception.Message)
            ? $"0x{exception.HResult:X8}" : $"0x{exception.HResult:X8}: {exception.Message}",
        int code => code == 0 ? "" : $"0x{code:X8}",
        uint code => code == 0 ? "" : $"0x{code:X8}",
        _ => value.ToString() ?? "",
    };

    private static async Task Reset()
    {
        lock (textLock) generation++;
        var previous = recognizer;
        recognizer = null;
        if (previous != null)
        {
            try { await previous.StopContinuousRecognitionAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(5)); }
            catch { /* Disposing below cancels a stopped or failed stream. */ }
            previous.Dispose();
        }
        provider?.Dispose();
        provider = null;
        lock (textLock)
        {
            completed.Clear();
            provisional = "";
        }
    }
}
