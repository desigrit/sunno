using System.Diagnostics;
using System.IO;
using System.Net.Http;
using Microsoft.UI.Xaml;
using Sunno.Services;
using Windows.ApplicationModel;

namespace Sunno;

public sealed partial class MainWindow
{
    private readonly HttpClient _updateHttp = AppUpdates.CreateClient();
    private CancellationTokenSource? _updateRequest;
    private AppRelease? _availableUpdate;
    private string? _downloadedUpdate;
    private static string UpdateDownloadFolder => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Sunno", "Updates");

    private static (Version Version, string Architecture)? InstalledUpdateIdentity()
    {
        try
        {
            var id = Package.Current.Id;
            var v = id.Version;
            return (new Version(v.Major, v.Minor, v.Build, v.Revision), id.Architecture.ToString().ToLowerInvariant());
        }
        catch (Exception ex) when (ex is InvalidOperationException or System.Runtime.InteropServices.COMException) { return null; }
    }

    private void RefreshUpdateAvailability()
    {
        if (InstalledUpdateIdentity() is null)
            AppUpdatePanel.Show("Check for updates", "Updates are available in the installed version of Sunno.", enabled: false);
    }

    private string UpdateActionLabel => _downloadedUpdate is not null ? "Install update"
        : _availableUpdate is not null ? "Download update" : "Check for updates";

    private async void OnUpdateAction(object sender, RoutedEventArgs e)
    {
        if (_updateRequest is not null) return;
        if (_downloadedUpdate is not null) await OpenUpdateInstallerAsync();
        else if (_availableUpdate is not null) await DownloadUpdateAsync();
        else await CheckForUpdatesAsync();
    }

    private async void OnCheckAppUpdates(object sender, RoutedEventArgs e) => await CheckForUpdatesAsync();
    private void OnCancelAppUpdate(object sender, RoutedEventArgs e) => _updateRequest?.Cancel();

    private async void OnViewAppReleases(object sender, RoutedEventArgs e)
    {
        try
        {
            if (!await Windows.System.Launcher.LaunchUriAsync(new Uri(AppUpdates.ReleasesUrl)))
                AppUpdatePanel.Show(UpdateActionLabel, "Windows could not open GitHub. Try your browser again.");
        }
        catch { AppUpdatePanel.Show(UpdateActionLabel, "Windows could not open GitHub. Try your browser again."); }
    }

    private async Task CheckForUpdatesAsync()
    {
        if (_updateRequest is not null || InstalledUpdateIdentity() is not { } installed) return;
        using var request = CancellationTokenSource.CreateLinkedTokenSource(_windowLifetime.Token);
        _updateRequest = request;
        AppUpdatePanel.Show("Checking...", "Checking GitHub releases...", busy: true);
        try
        {
            // Reclaim only this updater's cached packages already installed or superseded.
            await Task.Run(() => AppUpdates.ClearInstalledDownloads(UpdateDownloadFolder,
                installed.Version, installed.Architecture), request.Token);
            var result = await new AppUpdates(_updateHttp).CheckAsync(installed.Version, installed.Architecture, request.Token);
            request.Token.ThrowIfCancellationRequested();
            if (_availableUpdate != result.Update) _downloadedUpdate = null;
            _availableUpdate = result.Update;
            AppUpdatePanel.Show(UpdateActionLabel, result.Update is { } release
                ? _downloadedUpdate is not null ? ReadyMessage(release)
                    : $"Sunno {release.Version} is available ({release.Size / (1024 * 1024):N0} MB)."
                : result.HasAppReleases ? "You're up to date."
                : "No compatible Sunno installer is published on GitHub yet.");
        }
        catch (Exception ex)
        {
            if (!_windowLifetime.IsCancellationRequested)
                AppUpdatePanel.Show(UpdateActionLabel, request.IsCancellationRequested ? "Update check cancelled." : AppUpdates.FailureMessage(ex));
        }
        finally { _updateRequest = null; }
    }

    private static string ReadyMessage(AppRelease release) =>
        $"Sunno {release.Version} is ready. Updating closes Sunno and clears the current live transcript.";

    private async Task DownloadUpdateAsync()
    {
        if (_updateRequest is not null || _availableUpdate is not { } release || InstalledUpdateIdentity() is not { } installed) return;
        using var request = CancellationTokenSource.CreateLinkedTokenSource(_windowLifetime.Token);
        _updateRequest = request;
        AppUpdatePanel.Show("Downloading...", $"Downloading Sunno {release.Version}...", busy: true, progress: 0);
        try
        {
            var progress = new Progress<double>(value =>
            {
                if (ReferenceEquals(_updateRequest, request) && !request.IsCancellationRequested)
                    AppUpdatePanel.Show("Downloading...", $"Downloading Sunno {release.Version}: {value:0}%", busy: true, progress: value);
            });
            // No model, settings or recording directory is involved in an update download.
            var downloaded = await Task.Run(() => new AppUpdates(_updateHttp).DownloadAsync(
                release, installed.Version, installed.Architecture, UpdateDownloadFolder, progress, request.Token), request.Token);
            request.Token.ThrowIfCancellationRequested();
            _downloadedUpdate = downloaded;
            AppUpdatePanel.Show(UpdateActionLabel, ReadyMessage(release));
        }
        catch (Exception ex)
        {
            if (!_windowLifetime.IsCancellationRequested)
                AppUpdatePanel.Show(UpdateActionLabel, request.IsCancellationRequested ? "Download cancelled." : AppUpdates.FailureMessage(ex));
        }
        finally { _updateRequest = null; }
    }

    private async Task OpenUpdateInstallerAsync()
    {
        if (_downloadedUpdate is not { } path || _availableUpdate is not { } release || InstalledUpdateIdentity() is not { } installed) return;
        if (_recording)
        {
            AppUpdatePanel.Show("Install update", "Stop your recording before installing the update.");
            return;
        }
        using var request = CancellationTokenSource.CreateLinkedTokenSource(_windowLifetime.Token);
        _updateRequest = request;
        AppUpdatePanel.Show("Verifying...", "Verifying the downloaded update...", busy: true);
        try
        {
            await Task.Run(async () =>
            {
                // Recheck after waiting in Settings; don't hand a changed cache file to the installer.
                await using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 131072, true);
                var hash = await System.Security.Cryptography.SHA256.HashDataAsync(file, request.Token);
                if (file.Length != release.Size || !Convert.ToHexString(hash).Equals(release.Sha256, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("The downloaded update changed. Check for updates and download it again.");
                AppUpdates.ValidatePackage(path, release.Version, installed.Version, installed.Architecture);
            }, request.Token);
            request.Token.ThrowIfCancellationRequested();
            if (_recording) throw new InvalidOperationException("Stop your recording before installing the update.");
            // Hand off to the same native installer as double-clicking an MSIX. It owns
            // certificate trust, package validation, in-use prompts and the final result.
            Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
            AppUpdatePanel.Show("Install update", "Windows App Installer is open. Finish the update there, then open Sunno.");
        }
        catch (Exception ex)
        {
            if (!_windowLifetime.IsCancellationRequested)
            {
                if (ex is IOException or InvalidDataException) _downloadedUpdate = null;
                AppUpdatePanel.Show(UpdateActionLabel, request.IsCancellationRequested ? "Installation cancelled." : AppUpdates.FailureMessage(ex));
            }
        }
        finally { _updateRequest = null; }
    }
}
