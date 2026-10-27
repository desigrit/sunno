using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;
using Sunno.Services;

// Links the production services directly. Never calls AppSettings.Load or Save.
var checks = 0;
void Check(bool condition, string label)
{
    if (!condition) throw new Exception(label);
    checks++;
    Console.WriteLine($"PASS {label}");
}

using (var watcher = new AudioEndpointWatcher())
{
    watcher.Dispose();
    watcher.Dispose();
    Check(true, "native endpoint notifications register and dispose idempotently");
}

var legacy = new AppSettings { LoopbackDeviceIndex = 42, LoopbackDeviceName = "Old output" };
Check(legacy.InputTarget.Kind == "loopback" && !legacy.InputTarget.FollowDefault,
    "legacy system-audio choices retain their category");
Check(new AppSettings().InputTarget.FollowDefault, "first run follows the Windows microphone default");
legacy.RememberInput(new AudioInputTarget("loopback", "stable-output", "New output", 7, true));
var roundTrip = JsonSerializer.Deserialize<AppSettings>(JsonSerializer.Serialize(legacy))!;
Check(roundTrip.IsLoopback && roundTrip.InputTarget.FollowDefault && roundTrip.LoopbackDeviceIndex is null
    && roundTrip.InputEndpointId == "stable-output", "default mode persists separately from endpoint identity");

Check(new AppSettings().Appearance == AppearancePreference.System, "appearance defaults to System");
Check(!new AppSettings().SpeakerLabelsEnabled, "experimental speaker labels are opt-in on a fresh install");
Check(AppSettings.FromJson("{\"Model\":\"large-v3\"}")!.SpeakerLabelsEnabled,
    "an existing installation retains Automatic until the user changes it");
foreach (var enabled in new[] { false, true })
{
    var saved = new AppSettings { SpeakerLabelsEnabled = enabled };
    Check(AppSettings.FromJson(JsonSerializer.Serialize(saved))!.SpeakerLabelsEnabled == enabled,
        "Off and Automatic both persist across settings reloads");
}
var speakerLine = new Sunno.Models.CaptionLine
    { Text = "hello", SpeakerId = 1, SpeakerLabel = "Presenter", IsSelf = true, Clarity = 90, IsFinal = true };
Sunno.Models.CaptionLine.SpeakerLabelsEnabled = false;
Check(!speakerLine.HasSpeaker && !speakerLine.ShowClarity && !speakerLine.IsSelf
      && speakerLine.ToPlainText() == "hello", "Off hides labels, self dimming and copied speaker attribution");
Sunno.Models.CaptionLine.SpeakerLabelsEnabled = true;
Check(speakerLine.HasSpeaker && speakerLine.IsSelf && speakerLine.ShowClarity,
    "Automatic restores existing label metadata without changing words");
var colourChanged = false;
speakerLine.PropertyChanged += (_, e) => colourChanged |= e.PropertyName == "ColourIndex";
speakerLine.SpeakerId = 2;
Check(colourChanged && speakerLine.ColourIndex == 2, "speaker corrections refresh the existing caption color");
foreach (var preference in new[] { AppearancePreference.System, AppearancePreference.Light, AppearancePreference.Dark })
{
    var chosen = new AppSettings { Appearance = preference, Model = "moonshine-small", ForceCpu = true };
    var restored = JsonSerializer.Deserialize<AppSettings>(JsonSerializer.Serialize(chosen))!;
    Check(restored.Appearance == preference && restored.Model == chosen.Model && restored.ForceCpu,
        $"{preference} appearance survives serialization without changing speech preferences");
}
foreach (var json in new[] { "{\"Model\":\"moonshine-small\"}",
                             "{\"Appearance\":null,\"Model\":\"moonshine-small\"}",
                             "{\"Appearance\":\"future-theme\",\"Model\":\"moonshine-small\"}" })
{
    var restored = JsonSerializer.Deserialize<AppSettings>(json)!;
    Check(restored.Appearance == AppearancePreference.System && restored.Model == "moonshine-small",
        "missing, null or unknown appearance values preserve the existing model and fall back to System");
}
Check(AppearancePreference.Normalize(" DARK ") == AppearancePreference.Dark,
    "appearance settings tolerate casing and surrounding whitespace");
Check(AppearancePreference.IndexOf("system") == 0 && AppearancePreference.IndexOf("light") == 1
      && AppearancePreference.IndexOf("dark") == 2 && AppearancePreference.IndexOf("invalid") == 0,
    "appearance selector ordering is stable and invalid values are safe");
var contrastPalette = AppearancePalette.ForHighContrast(0xFFFFFFFF);
Check(contrastPalette.Good == contrastPalette.Mid && contrastPalette.Mid == contrastPalette.Low
      && contrastPalette.Good == 0xFFFFFFFF, "contrast themes use the system text color for every clarity status");

static double Luminance(uint rgb)
{
    static double Channel(uint channel)
    {
        var value = channel / 255.0;
        return value <= 0.04045 ? value / 12.92 : Math.Pow((value + 0.055) / 1.055, 2.4);
    }
    return 0.2126 * Channel((rgb >> 16) & 255) + 0.7152 * Channel((rgb >> 8) & 255)
           + 0.0722 * Channel(rgb & 255);
}
foreach (var dark in new[] { false, true })
{
    var palette = AppearancePalette.ForTheme(dark);
    var background = Luminance(dark ? 0xFF303030 : 0xFFE5E5E5);
    foreach (var foreground in new[] { palette.Good, palette.Mid, palette.Low,
                                      AppearancePalette.UncertainText(dark) })
    {
        var text = Luminance(foreground);
        var contrast = (Math.Max(text, background) + 0.05) / (Math.Min(text, background) + 0.05);
        Check(contrast >= 4.5, $"caption-status text meets 4.5:1 contrast in {(dark ? "dark" : "light")} appearance");
    }
}

foreach (var height in new[] { 120.0, 460.0, 760.0, 1200.0 })
{
    var listHeight = ModelPickerLayout.MaximumListHeight(height, 14, 44, 8);
    Check(listHeight >= 0, "model viewport never has a negative height");
    Check(listHeight + 44 + 8 <= (height - 14) * 0.75,
        "open model section preserves a quarter of the sidebar for speakers");
}
Check(ModelPickerLayout.MaximumListHeight(0, 14, 44, 8) == 0,
    "unloaded or hidden sidebar keeps the model viewport collapsed");
Check(ReadinessMessages.ForState("Ready") == "", "ready system model has no setup warning");
Check(ReadinessMessages.ForState("DisabledByUser").Contains("administrator"),
    "disabled AI models have a recovery action instead of another download");
Check(ReadinessMessages.ForState("CapabilityMissing").Contains("Sunno package"),
    "missing package capability points to the app update");
Check(ReadinessMessages.ForState("OSUpdateNeeded").Contains("Windows Update"),
    "old Windows versions have an actionable update message");
Check(ReadinessMessages.ForSetup("NotReady", "InProgress", "", "", "", false)
    .Contains("still installing"), "pending OS download is not reported as a setup failure");
var setupFailure = ReadinessMessages.ForSetup("NotReady", "Failure", "Download failed",
    "0x8000FFFF", "0x80072EE7", true);
Check(setupFailure.Contains("0x8000FFFF") && setupFailure.Contains("0x80072EE7")
      && setupFailure.Contains("Download failed"), "OS setup error details are preserved");

var builder = WebApplication.CreateBuilder();
builder.Logging.ClearProviders();
builder.WebHost.UseUrls("http://127.0.0.1:0");
await using var server = builder.Build();
server.UseWebSockets();
var received = 0;
var sockets = 0;
server.Run(async context =>
{
    using var socket = await context.WebSockets.AcceptWebSocketAsync();
    Interlocked.Increment(ref sockets);
    var buffer = new byte[16384];
    try
    {
        while (socket.State == WebSocketState.Open)
        {
            var result = await socket.ReceiveAsync(buffer, CancellationToken.None);
            if (result.MessageType == WebSocketMessageType.Close) break;
            using var doc = JsonDocument.Parse(buffer.AsMemory(0, result.Count));
            var root = doc.RootElement;
            if (root.GetProperty("cmd").GetString() == "set_speaker_labels")
            {
                foreach (var frame in new object[] {
                    new { type = "speaker_labels", mode = root.GetProperty("mode").GetString(), available = true, session = "test" },
                    new { type = "speaker_label", id = 42, started_at = 123.5, speaker_id = 3, speaker = "Presenter" }
                })
                    await socket.SendAsync(JsonSerializer.SerializeToUtf8Bytes(frame), WebSocketMessageType.Text,
                                           true, CancellationToken.None);
                continue;
            }
            if (root.GetProperty("cmd").GetString() != "set_input") continue;
            var selected = root.GetProperty("target");
            if (!selected.TryGetProperty("endpoint_id", out _)) throw new Exception("Incorrect wire property names");
            Interlocked.Increment(ref received);
            var message = JsonSerializer.SerializeToUtf8Bytes(new
            {
                type = "input", state = "ready", request_id = root.GetProperty("request_id").GetString(),
                target = selected, active = selected, wanted = true, running = true, committed = true,
                message = "Input ready: 麦克风"
            }, new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
            // Split inside a multibyte character to exercise real frame reassembly.
            var split = Array.IndexOf(message, (byte)0xE9);
            if (split < 0) split = message.Length / 2;
            else split++;
            await socket.SendAsync(message.AsMemory(0, split), WebSocketMessageType.Text, false, CancellationToken.None);
            await socket.SendAsync(message.AsMemory(split), WebSocketMessageType.Text, true, CancellationToken.None);
            if (received == 25)
            {
                await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "test reconnect", CancellationToken.None);
                break;
            }
        }
    }
    catch (WebSocketException) { }
    catch (OperationCanceledException) { }
});
await server.StartAsync();
var address = new Uri(server.Urls.Single());
await using var client = new CaptionClient("127.0.0.1", address.Port);
var acknowledgements = 0;
var connects = 0;
client.ConnectionChanged += connected => { if (connected) Interlocked.Increment(ref connects); };
client.Input += state =>
{
    if (state.Committed && state.Target?.EndpointId == "stable-input"
        && state.Message == "Input ready: 麦克风") Interlocked.Increment(ref acknowledgements);
};
var selection = new AudioInputTarget("microphone", "stable-input", "麦克风", 3, false);
Check(!await client.SetInputAsync(selection, "offline"), "disconnected sends report failure for replay");
client.Start();
async Task WaitFor(Func<bool> condition)
{
    var deadline = DateTime.UtcNow.AddSeconds(10);
    while (!condition() && DateTime.UtcNow < deadline) await Task.Delay(10);
    if (!condition()) throw new Exception("Transport test timed out");
}
await WaitFor(() => client.IsConnected);
var labelState = false;
var labelCorrection = false;
client.SpeakerLabelsState += (enabled, available, session) => labelState = enabled && available && session == "test";
client.SpeakerLabelChanged += (id, at, sid, label) => labelCorrection = id == 42 && at == 123.5 && sid == 3 && label == "Presenter";
Check(await client.SetSpeakerLabelsAsync(true), "speaker mode changes use the existing live connection");
await WaitFor(() => labelState && labelCorrection);
Check(labelState && labelCorrection, "speaker state and precise caption corrections survive the wire protocol");
var sends = await Task.WhenAll(Enumerable.Range(0, 25).Select(i => client.SetInputAsync(selection, $"request-{i}")));
Check(sends.All(sent => sent), "concurrent control sends are serialized successfully");
await WaitFor(() => acknowledgements == 25);
Check(received == 25, "all input requests arrive intact");
Check(acknowledgements == 25, "structured acknowledgements preserve fragmented Unicode");
await WaitFor(() => connects >= 2 && client.IsConnected);
Check(await client.SetInputAsync(selection, "after-reconnect"), "input commands work after reconnect");
await WaitFor(() => acknowledgements == 26);
Check(sockets == 2, "reconnect replaces the previous socket");
await client.DisposeAsync();
await server.StopAsync();
Console.WriteLine($"{checks} checks passed.");

namespace Sunno
{
    public static class App { public static void Trace(string message) { } }
}
