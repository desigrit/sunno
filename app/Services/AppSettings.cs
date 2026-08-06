using System.Text.Json;

namespace Sunno.Services;

/// <summary>
/// User choices that must survive a restart: which model was downloaded, and which
/// microphone to use.
///
/// Stored under LocalAppData rather than beside the executable, because an MSIX install
/// directory is read-only.
/// </summary>
public sealed class AppSettings
{
    public string Model { get; set; } = "large-v3";
    public int? DeviceIndex { get; set; }
    /// <summary>A WASAPI output endpoint to caption instead of a microphone.</summary>
    public int? LoopbackDeviceIndex { get; set; }
    /// <summary>Stable WASAPI endpoint id. The index above remains for settings migration.</summary>
    public string? LoopbackDeviceId { get; set; }

    /// <summary>
    /// The name of the chosen capture device, and what makes the choice survive a restart.
    ///
    /// The index above cannot be trusted on its own. PortAudio numbers devices by enumeration
    /// order, so the numbers move whenever the set of audio devices changes: a Bluetooth headset
    /// connecting, a monitor waking, a USB mic being plugged in. Measured on one machine across
    /// two launches, the same Umik-1 moved from index 30 to 27, and index 26 stopped meaning
    /// "Microphone (2- Logitech BRIO)" and started meaning "Headset (R-Phonak hearing aid)".
    ///
    /// So the index is a fast path and the name is the truth. The app still launches on the
    /// index, because capture has to start before the device list can be fetched from the
    /// backend that serves it, and then checks the name once the list arrives. See
    /// MainWindow.ValidateRememberedDevice.
    ///
    /// Stored as the cleaned display name, the same string the picker shows, so it can be
    /// compared with the picker's own matching rules rather than raw PortAudio spelling.
    /// </summary>
    public string? DeviceName { get; set; }

    /// <summary>As <see cref="DeviceName"/>, for the system-audio endpoint.</summary>
    public string? LoopbackDeviceName { get; set; }

    public string? Vocabulary { get; set; }
    public double CaptionFontSize { get; set; } = 26;
    public bool AlwaysOnTop { get; set; } = true;

    /// <summary>Whether the clarity badge appears on the user's own lines.</summary>
    public bool ShowClarity { get; set; } = true;

    /// <summary>
    /// Pin the engine to the processor. Off means "let it choose", which prefers the graphics
    /// card. Exists as a way back in when a driver update stops CUDA loading: without it the
    /// app is dead with no recourse from inside the UI.
    /// </summary>
    public bool ForceCpu { get; set; }

    private static string FilePath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Sunno", "settings.json");

    public static AppSettings Load()
    {
        try
        {
            if (File.Exists(FilePath))
            {
                var json = File.ReadAllText(FilePath);
                return JsonSerializer.Deserialize<AppSettings>(json) ?? new AppSettings();
            }
        }
        catch
        {
            // A corrupt settings file must not stop the app starting.
        }
        return new AppSettings();
    }

    public void Save()
    {
        try
        {
            var dir = Path.GetDirectoryName(FilePath)!;
            Directory.CreateDirectory(dir);

            // Write to a sibling temp file and move it into place, so an interrupted write
            // cannot leave a half-written settings.json behind. Load() treats a corrupt file as
            // "no settings", so a torn write would silently reset the user's device and model
            // choice with nothing on screen to explain it.
            //
            // File.Move(overwrite) rather than File.Replace: Replace throws
            // FileNotFoundException when the destination does not exist yet, and plenty of
            // first saves happen on a machine that has no settings.json - the first caption
            // size change, the first device choice. The catch below would have swallowed that,
            // and the setting would have failed to persist on every new install, silently.
            var temp = FilePath + ".tmp";
            File.WriteAllText(temp,
                JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
            File.Move(temp, FilePath, overwrite: true);
        }
        catch
        {
            // Losing a preference is not worth surfacing an error for.
        }
    }
}
