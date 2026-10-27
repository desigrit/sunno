using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Sunno.Services;

var passed = 0;
void Check(bool condition, string label)
{
    if (!condition) throw new Exception(label);
    Console.WriteLine("PASS " + label);
    passed++;
}
async Task Reject(Func<Task> action, string label)
{
    try { await action(); } catch (Exception ex) when (ex is InvalidDataException or HttpRequestException or OperationCanceledException or InvalidOperationException)
    { Check(true, label); return; }
    throw new Exception("Did not reject: " + label);
}
var current = new Version(1, 0, 90, 0);
var next = new Version(1, 0, 91, 0);
object Release(string version, string? name = null, bool preview = false, bool draft = false, string? url = null, string? digest = null) => new
{
    tag_name = version, prerelease = preview, draft,
    assets = new[] { new { name = name ?? $"Sunno_{version.TrimStart('v')}_x64.msix", size = 1234,
        browser_download_url = url ?? $"https://github.com/desigrit/sunno/releases/download/{version}/Sunno.msix",
        digest = digest ?? "sha256:" + new string('a', 64) } },
};
AppUpdateCheck Select(params object[] releases) => AppUpdates.SelectRelease(JsonSerializer.SerializeToUtf8Bytes(releases), current, "x64");
Check(Select(Release("cuda-12.9.2", "cublas.dll.xz")).Update is null, "CUDA releases are not app updates");
Check(!Select(Release("cuda-12.9.2", "cublas.dll.xz")).HasAppReleases, "no app releases is distinct from up to date");
Check(Select(Release("v1.0.90.0")).Update is null && Select(Release("v1.0.90.0")).HasAppReleases, "current version is up to date");
Check(Select(Release("v1.0.89.0")).Update is null, "older releases cannot downgrade");
Check(Select(Release("v1.0.100.0"), Release("v1.0.99.0")).Update?.Version == new Version(1,0,100,0), "versions are compared numerically");
Check(Select(Release("v1.0.91.0", preview: true), Release("v1.0.92.0", draft: true)).Update is null, "draft and preview releases are skipped");
Check(Select(Release("v1.0.91.0", "Sunno_1.0.91.0_arm64.msix")).Update is null, "wrong architecture is excluded");
Check(Select(Release("v1.0.91.0", digest: "sha256:bad")).Update is null, "malformed digests are excluded");
Check(Select(Release("v1.0.91.0", url: "https://example.com/Sunno.msix")).Update is null, "unrelated download hosts are excluded");
Check(Select(Release("v1.0.91.0", url: "https://github.com/other/app/releases/download/a/Sunno.msix")).Update is null, "another repository cannot supply an update");
Check(!AppUpdates.IsReleaseAsset(new Uri("http://github.com/desigrit/sunno/releases/download/a/a.msix")), "HTTP downloads are forbidden");
Check(!AppUpdates.TryVersion("v1.0.91-rc1", out _) && !AppUpdates.TryVersion("1.0.99999.0", out _), "non-package versions are rejected");
Check(AppUpdates.TryVersion("v1.0.91", out var shortVersion) && shortVersion == next, "three-part versions normalize to the package version");

byte[] Package(string name = AppUpdates.PackageName, string publisher = AppUpdates.Publisher, string architecture = "x64", string version = "1.0.91.0", bool signed = true)
{
    using var buffer = new MemoryStream();
    using (var zip = new ZipArchive(buffer, ZipArchiveMode.Create, true))
    {
        using (var text = new StreamWriter(zip.CreateEntry("AppxManifest.xml").Open()))
            text.Write($"<Package xmlns=\"http://schemas.microsoft.com/appx/manifest/foundation/windows10\"><Identity Name=\"{name}\" Publisher=\"{publisher}\" Version=\"{version}\" ProcessorArchitecture=\"{architecture}\" /></Package>");
        // Presence is a metadata check only. The native installer must verify real signatures.
        if (signed) zip.CreateEntry("AppxSignature.p7x");
    }
    return buffer.ToArray();
}
var payload = Package();
AppRelease Asset(byte[] bytes) => new(next, new Uri("https://github.com/desigrit/sunno/releases/download/v1.0.91.0/Sunno_1.0.91.0_x64.msix"), bytes.Length, Convert.ToHexString(SHA256.HashData(bytes)));
var folder = Path.Combine(Path.GetTempPath(), "SunnoUpdateTests-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(folder);
try
{
    using var handler = new StubHandler(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(payload) });
    using var http = new HttpClient(handler);
    var service = new AppUpdates(http);
    var path = await service.DownloadAsync(Asset(payload), current, "x64", folder, null, default);
    Check(File.ReadAllBytes(path).SequenceEqual(payload) && !File.Exists(path + ".partial"), "verified download is complete and partial is removed");
    await Reject(() => service.DownloadAsync(Asset(payload) with { Sha256 = new string('0', 64) }, current, "x64", folder, null, default), "bad hash is rejected");
    Check(!File.Exists(path + ".partial"), "failed download leaves no partial file");
    await Reject(() => service.DownloadAsync(Asset(payload) with { Size = payload.Length + 1 }, current, "x64", folder, null, default), "truncated download is rejected");
    foreach (var (bytes, label) in new[] {
        (Package(name: "Other.App"), "wrong package identity"),
        (Package(publisher: "CN=SomeoneElse"), "wrong publisher"),
        (Package(architecture: "arm64"), "wrong package architecture"),
        (Package(version: "1.0.89.0"), "older package behind a newer release tag"),
        (Package(signed: false), "unsigned package") })
    {
        using var invalidHttp = new HttpClient(new StubHandler(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) }));
        await Reject(() => new AppUpdates(invalidHttp).DownloadAsync(Asset(bytes), current, "x64", folder, null, default), label + " is rejected");
    }
    using var cancelled = new CancellationTokenSource();
    cancelled.Cancel();
    await Reject(() => service.DownloadAsync(Asset(payload), current, "x64", folder, null, cancelled.Token), "cancelled download cannot produce a new installer");
    using var redirectHttp = new HttpClient(new StubHandler(_ => new HttpResponseMessage(HttpStatusCode.Redirect)
        { Headers = { Location = new Uri("https://example.com/payload.msix") } }));
    await Reject(() => new AppUpdates(redirectHttp).DownloadAsync(Asset(payload), current, "x64", folder, null, default), "unexpected redirect is rejected before following it");
    using var allowedHttp = new HttpClient(new StubHandler(uri => uri.Host == "github.com"
        ? new HttpResponseMessage(HttpStatusCode.Redirect) { Headers = { Location = new Uri("https://release-assets.githubusercontent.com/asset") } }
        : new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(payload) }));
    Check(File.Exists(await new AppUpdates(allowedHttp).DownloadAsync(Asset(payload), current, "x64", folder, null, default)), "GitHub's release CDN redirect works");
    using var offlineHttp = new HttpClient(new StubHandler(_ => throw new HttpRequestException("offline")));
    await Reject(() => new AppUpdates(offlineHttp).CheckAsync(current, "x64", default), "offline check is a retryable failure, not up to date");
    using var limitHttp = new HttpClient(new StubHandler(_ => new HttpResponseMessage(HttpStatusCode.Forbidden)));
    await Reject(() => new AppUpdates(limitHttp).CheckAsync(current, "x64", default), "GitHub rate limiting is explicit");
    var oldPath = Path.Combine(folder, "Sunno_1.0.89.0_x64.msix");
    var unrelated = Path.Combine(folder, "keep.msix");
    File.WriteAllText(oldPath, "cache"); File.WriteAllText(unrelated, "not updater owned");
    AppUpdates.ClearInstalledDownloads(folder, current, "x64");
    Check(!File.Exists(oldPath) && File.Exists(unrelated) && File.Exists(path), "cleanup preserves unrelated files and pending newer update");
    if (args.Contains("--live"))
    {
        using var live = AppUpdates.CreateClient();
        var result = await new AppUpdates(live).CheckAsync(current, "x64", default);
        Console.WriteLine($"Live GitHub check: app releases={result.HasAppReleases}, update={result.Update?.Version.ToString() ?? "none"}");
    }
    if (args.FirstOrDefault(a => a.EndsWith(".msix", StringComparison.OrdinalIgnoreCase)) is { } realPackage)
    {
        AppUpdates.ValidatePackage(realPackage, new Version(1,0,89,0), new Version(1,0,88,0), "x64");
        Check(true, "production signed MSIX metadata is accepted as an update from 1.0.88");
    }
    Console.WriteLine($"{passed} update checks passed.");
}
finally
{
    // The unique test directory was created above and is never user supplied.
    Directory.Delete(folder, recursive: true);
}

sealed class StubHandler(Func<Uri, HttpResponseMessage> respond) : HttpMessageHandler
{
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token)
    { token.ThrowIfCancellationRequested(); return Task.FromResult(respond(request.RequestUri!)); }
}
