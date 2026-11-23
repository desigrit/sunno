using System.IO;
using Windows.Storage;

namespace Sunno.Services;

public sealed partial class AppUpdates
{
    // App Installer runs outside Sunno's MSIX filesystem view. Give both processes
    // the physical cache path so the file we verified is the file Windows can open.
    internal static string DownloadDirectory => Path.Combine(ApplicationData.Current.LocalCacheFolder.Path, "Updates");

    internal static void ClearInstalledDownloads(Version installed, string architecture)
    {
        ClearInstalledDownloads(DownloadDirectory, installed, architecture);
        // Older builds saved packages via the redirected LocalApplicationData path.
        var previousDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Sunno", "Updates");
        ClearInstalledDownloads(previousDirectory, installed, architecture);
    }
}
