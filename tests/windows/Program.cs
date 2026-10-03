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
