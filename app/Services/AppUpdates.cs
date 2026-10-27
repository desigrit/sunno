using System.IO;
using System.IO.Compression;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml;
using System.Xml.Linq;

namespace Sunno.Services;

public sealed record AppRelease(Version Version, Uri DownloadUrl, long Size, string Sha256);
public sealed record AppUpdateCheck(AppRelease? Update, bool HasAppReleases);

/// <summary>Explicit, anonymous checks of this app's stable GitHub releases only.</summary>
public sealed class AppUpdates(HttpClient client)
{
    public const string PackageName = "DesigritLabs.Sunno";
    public const string Publisher = "CN=A2015C41-8111-42CA-8A27-273B3309C099";
    public const string ReleasesUrl = "https://github.com/desigrit/sunno/releases";
    private const string ApiUrl = "https://api.github.com/repos/desigrit/sunno/releases?per_page=100";
    private const long MaxPackageSize = 2L * 1024 * 1024 * 1024;

    public static HttpClient CreateClient()
    {
        var http = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false, UseCookies = false })
        { Timeout = Timeout.InfiniteTimeSpan };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("Sunno-Updater/1.0");
        return http;
    }

    public async Task<AppUpdateCheck> CheckAsync(Version currentVersion, string architecture, CancellationToken token)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(token);
        timeout.CancelAfter(TimeSpan.FromSeconds(20));
        using var response = await client.GetAsync(ApiUrl, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
        if (response.StatusCode is HttpStatusCode.Forbidden or HttpStatusCode.TooManyRequests)
            throw new InvalidOperationException("GitHub is limiting update checks. Try again later.");
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token);
        // The public endpoint is small; refuse an unexpectedly large response.
        using var buffer = new MemoryStream();
        var chunk = new byte[32768];
        int count;
        while ((count = await stream.ReadAsync(chunk, timeout.Token)) > 0)
        {
            if (buffer.Length + count > 4 * 1024 * 1024) throw new InvalidDataException("GitHub returned an unexpected update response.");
            buffer.Write(chunk, 0, count);
        }
        return SelectRelease(buffer.ToArray(), currentVersion, architecture);
    }

    internal static AppUpdateCheck SelectRelease(byte[] json, Version current, string architecture)
    {
        using var document = JsonDocument.Parse(json);
        var releases = new List<AppRelease>();
        if (document.RootElement.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException("GitHub returned an unexpected update response.");
        foreach (var release in document.RootElement.EnumerateArray())
        {
            if (release.ValueKind != JsonValueKind.Object
                || !release.TryGetProperty("draft", out var draft) || draft.ValueKind != JsonValueKind.False
                || !release.TryGetProperty("prerelease", out var preview) || preview.ValueKind != JsonValueKind.False
                || !release.TryGetProperty("tag_name", out var tag) || tag.ValueKind != JsonValueKind.String
                || !TryVersion(tag.GetString(), out var version)
                || !release.TryGetProperty("assets", out var assets) || assets.ValueKind != JsonValueKind.Array) continue;
            foreach (var asset in assets.EnumerateArray())
            {
                if (asset.ValueKind != JsonValueKind.Object
                    || !asset.TryGetProperty("name", out var name) || name.ValueKind != JsonValueKind.String) continue;
                // Architecture is explicit. Never offer an ARM package to the x64 shell on Snapdragon.
                var expectedName = $"Sunno_{version}_{architecture.ToLowerInvariant()}.msix";
                if (!string.Equals(name.GetString(), expectedName, StringComparison.OrdinalIgnoreCase)) continue;
                if (!asset.TryGetProperty("browser_download_url", out var url) || url.ValueKind != JsonValueKind.String
                    || !Uri.TryCreate(url.GetString(), UriKind.Absolute, out var uri) || !IsReleaseAsset(uri)
                    || !asset.TryGetProperty("size", out var size) || !size.TryGetInt64(out var length)
                    || length <= 0 || length > MaxPackageSize
                    || !asset.TryGetProperty("digest", out var digest) || digest.ValueKind != JsonValueKind.String) continue;
                var hash = digest.GetString()!;
                if (!Regex.IsMatch(hash, "^sha256:[0-9a-fA-F]{64}$", RegexOptions.CultureInvariant)) continue;
                releases.Add(new(version, uri, length, hash[7..]));
            }
        }
        return new(releases.Where(r => r.Version > current).OrderByDescending(r => r.Version).FirstOrDefault(), releases.Count > 0);
    }

    internal static bool TryVersion(string? text, out Version version)
    {
        version = new Version(0, 0, 0, 0);
        if (string.IsNullOrWhiteSpace(text)) return false;
        text = text.TrimStart('v', 'V');
        if (!Regex.IsMatch(text, "^[0-9]+\\.[0-9]+\\.[0-9]+(?:\\.[0-9]+)?$", RegexOptions.CultureInvariant)
            || !Version.TryParse(text, out var parsed)) return false;
        version = new(parsed.Major, parsed.Minor, parsed.Build, Math.Max(0, parsed.Revision));
        return version.Major <= ushort.MaxValue && version.Minor <= ushort.MaxValue
            && version.Build <= ushort.MaxValue && version.Revision <= ushort.MaxValue;
    }

    internal static bool IsReleaseAsset(Uri uri) => uri.Scheme == "https" && uri.IsDefaultPort
        && uri.Host.Equals("github.com", StringComparison.OrdinalIgnoreCase) && uri.UserInfo.Length == 0
        && uri.AbsolutePath.StartsWith("/desigrit/sunno/releases/download/", StringComparison.Ordinal);

    private static bool IsDownloadRedirect(Uri uri) => IsReleaseAsset(uri)
        || (uri.Scheme == "https" && uri.IsDefaultPort && uri.UserInfo.Length == 0
            && uri.Host.Equals("release-assets.githubusercontent.com", StringComparison.OrdinalIgnoreCase));

    public async Task<string> DownloadAsync(AppRelease release, Version current, string architecture,
        string directory, IProgress<double>? progress, CancellationToken token)
    {
        if (!IsReleaseAsset(release.DownloadUrl) || release.Size <= 0 || release.Size > MaxPackageSize
            || !Regex.IsMatch(release.Sha256, "^[0-9a-fA-F]{64}$", RegexOptions.CultureInvariant))
            throw new InvalidDataException("The update download information is invalid. Check for updates again.");
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, $"Sunno_{release.Version}_{architecture.ToLowerInvariant()}.msix");
        var partial = path + ".partial";
        try
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(token);
            timeout.CancelAfter(TimeSpan.FromMinutes(30));
            using var response = await OpenDownloadAsync(release.DownloadUrl, timeout.Token);
            if (response.Content.Headers.ContentLength is long size && size != release.Size)
                throw new InvalidDataException("The update size changed. Check for updates again.");
            await using (var input = await response.Content.ReadAsStreamAsync(timeout.Token))
            await using (var output = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, 131072, true))
            using (var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
            {
                var buffer = new byte[131072];
                long received = 0;
                var lastPercent = -1;
                int count;
                while ((count = await input.ReadAsync(buffer, timeout.Token)) > 0)
                {
                    received += count;
                    if (received > release.Size) throw new InvalidDataException("The update is larger than expected. Download it again.");
                    hash.AppendData(buffer, 0, count);
                    await output.WriteAsync(buffer.AsMemory(0, count), timeout.Token);
                    var percent = (int)(100 * received / release.Size);
                    if (percent != lastPercent) { lastPercent = percent; progress?.Report(percent); }
                }
                if (received != release.Size || !Convert.ToHexString(hash.GetHashAndReset()).Equals(release.Sha256, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("The update download could not be verified. Download it again.");
            }
            ValidatePackage(partial, release.Version, current, architecture);
            token.ThrowIfCancellationRequested();
            File.Move(partial, path, overwrite: true);
            return path;
        }
        finally
        {
            // Only the exact partial file owned by this download is removed.
            if (File.Exists(partial)) File.Delete(partial);
        }
    }

    private async Task<HttpResponseMessage> OpenDownloadAsync(Uri uri, CancellationToken token)
    {
        for (var redirects = 0; redirects < 5; redirects++)
        {
            if (!IsDownloadRedirect(uri)) throw new InvalidDataException("GitHub redirected the update to an unexpected location.");
            var response = await client.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, token);
            if ((int)response.StatusCode is >= 300 and < 400)
            {
                var next = response.Headers.Location;
                response.Dispose();
                if (next is null) throw new InvalidDataException("GitHub did not provide an update download location.");
                uri = next.IsAbsoluteUri ? next : new Uri(uri, next);
                continue;
            }
            if (response.IsSuccessStatusCode) return response;
            var status = response.StatusCode;
            response.Dispose();
            throw new HttpRequestException("GitHub could not provide the update download.", null, status);
        }
        throw new InvalidDataException("GitHub redirected the update too many times. Try again later.");
    }

    internal static void ValidatePackage(string path, Version expected, Version current, string architecture)
    {
        using var archive = ZipFile.OpenRead(path);
        var manifest = archive.GetEntry("AppxManifest.xml")
            ?? throw new InvalidDataException("The download is not a Windows app package.");
        if (manifest.Length > 1024 * 1024 || archive.GetEntry("AppxSignature.p7x") is null)
            throw new InvalidDataException("The update package is invalid or unsigned.");
        using var stream = manifest.Open();
        using var reader = XmlReader.Create(stream, new XmlReaderSettings
        { DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null, MaxCharactersInDocument = 1024 * 1024 });
        var xml = XDocument.Load(reader);
        XNamespace ns = "http://schemas.microsoft.com/appx/manifest/foundation/windows10";
        var identity = xml.Root?.Element(ns + "Identity");
        if ((string?)identity?.Attribute("Name") != PackageName
            || (string?)identity?.Attribute("Publisher") != Publisher
            || !string.Equals((string?)identity?.Attribute("ProcessorArchitecture"), architecture, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("This update does not match your Sunno installation.");
        if (!TryVersion((string?)identity?.Attribute("Version"), out var version) || version != expected || version <= current)
            throw new InvalidDataException("This package is not the expected newer Sunno version. Check for updates again.");
        // This checks package metadata, not trust. Windows App Installer performs signature
        // and certificate verification before deployment; Sunno never imports certificates.
    }

    internal static void ClearInstalledDownloads(string directory, Version installed, string architecture)
    {
        if (!Directory.Exists(directory)) return;
        foreach (var file in Directory.EnumerateFiles(directory, "Sunno_*.msix", SearchOption.TopDirectoryOnly))
        {
            var match = Regex.Match(Path.GetFileName(file), "^Sunno_([0-9.]+)_" + Regex.Escape(architecture) + "\\.msix$",
                RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);
            if (!match.Success || !TryVersion(match.Groups[1].Value, out var version) || version > installed) continue;
            try { File.Delete(file); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    public static string FailureMessage(Exception error) => error switch
    {
        OperationCanceledException => "The update request timed out. Try again.",
        HttpRequestException => "Could not reach GitHub. Check your connection and try again.",
        UnauthorizedAccessException => "Windows blocked access to the update folder. Check its permissions and try again.",
        InvalidDataException or InvalidOperationException => error.Message,
        IOException => "Could not save the update. Check free disk space and try again.",
        System.ComponentModel.Win32Exception => "Windows could not open the installer. Install or repair App Installer, then try again.",
        _ => "Could not finish the update request. Try again or open GitHub releases.",
    };
}
