using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Sunno.Models;

namespace Sunno.Services;

public sealed record CaptionEvent(
    string Type, int Id, string? Text, int? SpeakerId, string? Speaker,
    int? Clarity, double LatencyMs,
    double? StartedAt = null, IReadOnlyList<CaptionWord>? Words = null);

public sealed record StatusEvent(string State, bool? Running, string? Model, string? Device,
                                 string? Message, string? Code = null);

public sealed record LevelEvent(double Db, bool Speaking);

/// <summary>A model offered during first-run setup.</summary>
public sealed record ModelOption(
    string Id, string Name, string Detail, int ApproxMb, string Languages, bool Available,
    int LagMs = 0, bool Responsive = true);

public sealed record DownloadProgressEvent(string Model, long Downloaded, long Total, double Percent);

/// <summary>
/// WebSocket client for the Python captioning backend.
///
/// The same protocol serves this XAML app and any remote browser, which is what keeps the
/// "show captions on a handheld over WiFi" path free.
/// </summary>
public sealed class CaptionClient : IAsyncDisposable
{
    private readonly Uri _uri;
    private ClientWebSocket? _socket;
    private CancellationTokenSource? _cts;
    private Task? _pump;

    public CaptionClient(string host = "127.0.0.1", int port = 8766)
        => _uri = new Uri($"ws://{host}:{port}");

    public event Action<CaptionEvent>? Partial;
    public event Action<CaptionEvent>? Final;
    public event Action<int>? Discarded;
    public event Action<StatusEvent>? Status;
    public event Action<LevelEvent>? Level;
    public event Action<IReadOnlyList<SpeakerInfo>>? Roster;
    /// <summary>Two speakers were folded into one: captions tagged with the first id belong
    /// to the second now. Raised immediately before the roster that reflects the merge.</summary>
    public event Action<int, int>? SpeakersMerged;
    /// <summary>A speaker was forgotten. Captions tagged with that id should fall back to the
    /// supplied generic label. Raised immediately before the roster that reflects the removal.</summary>
    public event Action<int, string>? SpeakerDeleted;
    public event Action<bool>? ConnectionChanged;
    public event Action<IReadOnlyList<ModelOption>>? ModelRequired;
    /// <summary>
    /// The catalogue, the model currently selected, and the compute device the engine resolved.
    ///
    /// The device string here is "cuda" or "cpu" — the backend sends settings.device on this
    /// frame. Do not take it from the status frame instead: that one carries the *audio* device
    /// name, and reading it as a compute device once put a user's hearing aid into a diagnostics
    /// report.
    /// </summary>
    public event Action<string, string?, IReadOnlyList<ModelOption>>? ModelCatalog;
    public event Action<DownloadProgressEvent>? DownloadProgress;
    public event Action<string>? DownloadComplete;
    public event Action<string>? DownloadFailed;

    public bool IsConnected => _socket?.State == WebSocketState.Open;

    public void Start()
    {
        _cts = new CancellationTokenSource();
        _pump = Task.Run(() => RunAsync(_cts.Token));
    }

    /// <summary>Reconnects with backoff, so the UI survives a backend restart.</summary>
    private async Task RunAsync(CancellationToken token)
    {
        var delay = TimeSpan.FromMilliseconds(400);
        while (!token.IsCancellationRequested)
        {
            try
            {
                _socket = new ClientWebSocket();
                await _socket.ConnectAsync(_uri, token);
                ConnectionChanged?.Invoke(true);
                delay = TimeSpan.FromMilliseconds(400);
                await ReceiveLoopAsync(_socket, token);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch
            {
                // Backend not up yet, or it dropped. Fall through and retry.
            }

            ConnectionChanged?.Invoke(false);
            try { await Task.Delay(delay, token); }
            catch (OperationCanceledException) { return; }
            delay = TimeSpan.FromMilliseconds(Math.Min(delay.TotalMilliseconds * 2, 5000));
        }
    }

    private async Task ReceiveLoopAsync(ClientWebSocket socket, CancellationToken token)
    {
        var buffer = new byte[64 * 1024];
        var builder = new StringBuilder();

        while (socket.State == WebSocketState.Open && !token.IsCancellationRequested)
        {
            builder.Clear();
            WebSocketReceiveResult result;
            do
            {
                result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
                if (result.MessageType == WebSocketMessageType.Close) return;
                builder.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
            }
            while (!result.EndOfMessage);

            try { Dispatch(builder.ToString()); }
            catch (JsonException) { /* skip a malformed frame rather than drop the socket */ }
        }
    }

    private void Dispatch(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var type = root.GetProperty("type").GetString();

        switch (type)
        {
            case "partial":
            case "final":
            {
                var ev = new CaptionEvent(
                    type!, GetInt(root, "id") ?? 0, GetString(root, "text"),
                    GetInt(root, "speaker_id"), GetString(root, "speaker"),
                    GetInt(root, "clarity"), GetDouble(root, "latency_ms") ?? 0,
                    GetDouble(root, "started_at"), ParseWords(root));
                if (type == "partial") Partial?.Invoke(ev); else Final?.Invoke(ev);
                break;
            }
            case "discard":
                Discarded?.Invoke(GetInt(root, "id") ?? 0);
                break;
            case "level":
                Level?.Invoke(new LevelEvent(GetDouble(root, "db") ?? -60,
                                             GetBool(root, "speaking") ?? false));
                break;
            case "status":
                Status?.Invoke(new StatusEvent(
                    GetString(root, "state") ?? "unknown", GetBool(root, "running"),
                    GetString(root, "model"), GetString(root, "device"), null));
                break;
            case "error":
                Status?.Invoke(new StatusEvent("error", GetBool(root, "running"), null, null,
                                               GetString(root, "message"),
                                               GetString(root, "code")));
                break;
            case "roster":
            {
                var list = new List<SpeakerInfo>();
                if (root.TryGetProperty("speakers", out var speakers))
                {
                    foreach (var s in speakers.EnumerateArray())
                    {
                        list.Add(new SpeakerInfo(
                            GetInt(s, "id") ?? 0,
                            GetString(s, "label") ?? "Speaker",
                            GetBool(s, "named") ?? false,
                            GetBool(s, "is_self") ?? false));
                    }
                }
                Roster?.Invoke(list);
                break;
            }
            case "speaker_merged":
            {
                // Must be handled before the roster frame that follows it, and it is: frames
                // are dispatched one at a time as they are read off the socket.
                var from = GetInt(root, "from");
                var into = GetInt(root, "into");
                if (from is int f && into is int t) SpeakersMerged?.Invoke(f, t);
                break;
            }
            case "speaker_deleted":
            {
                var id = GetInt(root, "id");
                var label = GetString(root, "label");
                if (id is int d && !string.IsNullOrEmpty(label)) SpeakerDeleted?.Invoke(d, label);
                break;
            }
            case "model_required":
                ModelRequired?.Invoke(ParseCatalog(root));
                break;
            case "model_catalog":
                ModelCatalog?.Invoke(GetString(root, "current") ?? "",
                                     GetString(root, "device"), ParseCatalog(root));
                break;
            case "download_progress":
                DownloadProgress?.Invoke(new DownloadProgressEvent(
                    GetString(root, "model") ?? "",
                    GetLong(root, "downloaded") ?? 0,
                    GetLong(root, "total") ?? 0,
                    GetDouble(root, "percent") ?? 0));
                break;
            case "download_complete":
                DownloadComplete?.Invoke(GetString(root, "model") ?? "");
                break;
            case "download_failed":
                DownloadFailed?.Invoke(GetString(root, "message") ?? "Download failed.");
                break;
        }
    }

    public Task DownloadModelAsync(string model) =>
        SendAsync(new { cmd = "download_model", model });

    public Task RequestModelsAsync() => SendAsync(new { cmd = "list_models" });

    /// <summary>
    /// Per-word confidence, present on finals only. Materialised into records here rather than
    /// held as JsonElements: those are views over the JsonDocument's pooled buffer and throw
    /// once it is disposed, which happens before the UI thread ever sees them.
    /// </summary>
    private static IReadOnlyList<CaptionWord>? ParseWords(JsonElement root)
    {
        if (!root.TryGetProperty("words", out var words) ||
            words.ValueKind != JsonValueKind.Array) return null;

        var list = new List<CaptionWord>();
        foreach (var w in words.EnumerateArray())
        {
            var text = GetString(w, "t");
            if (text is null) continue;
            list.Add(new CaptionWord(text, GetDouble(w, "p") ?? 1.0));
        }
        return list.Count == 0 ? null : list;
    }

    /// <summary>
    /// Both model_required and model_catalog carry the same catalogue shape; the difference is
    /// only whether the backend is blocked waiting for a choice.
    /// </summary>
    private static IReadOnlyList<ModelOption> ParseCatalog(JsonElement root)
    {
        var options = new List<ModelOption>();
        if (!root.TryGetProperty("catalog", out var catalog)) return options;

        foreach (var m in catalog.EnumerateArray())
        {
            options.Add(new ModelOption(
                GetString(m, "id") ?? "",
                GetString(m, "name") ?? "",
                GetString(m, "detail") ?? "",
                GetInt(m, "approx_mb") ?? 0,
                GetString(m, "languages") ?? "",
                GetBool(m, "available") ?? false,
                GetInt(m, "lag_ms") ?? 0,
                GetBool(m, "responsive") ?? true));
        }
        return options;
    }

    public Task ToggleAsync() => SendAsync(new { cmd = "toggle" });
    public Task StartCaptureAsync() => SendAsync(new { cmd = "start" });
    public Task StopCaptureAsync() => SendAsync(new { cmd = "stop" });

    public Task RenameSpeakerAsync(int id, string name) =>
        SendAsync(new { cmd = "rename_speaker", id, name });

    public Task SetSelfAsync(int id, bool value) =>
        SendAsync(new { cmd = "set_self", id, value });

    public Task MergeSpeakersAsync(int source, int target) =>
        SendAsync(new { cmd = "merge_speakers", source, target });

    public Task DeleteSpeakerAsync(int id) =>
        SendAsync(new { cmd = "delete_speaker", id });

    private async Task SendAsync(object payload)
    {
        if (_socket is not { State: WebSocketState.Open }) return;
        var bytes = JsonSerializer.SerializeToUtf8Bytes(payload);
        await _socket.SendAsync(bytes, WebSocketMessageType.Text, true, CancellationToken.None);
    }

    private static string? GetString(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static int? GetInt(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;

    private static double? GetDouble(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;

    private static long? GetLong(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt64() : null;

    private static bool? GetBool(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? v.GetBoolean() : null;

    public async ValueTask DisposeAsync()
    {
        _cts?.Cancel();
        if (_pump is not null)
        {
            try { await _pump; } catch { /* shutting down */ }
        }
        _socket?.Dispose();
        _cts?.Dispose();
    }
}
