using System.Text.Json;
using System.Text.Json.Serialization;

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

    /// <summary>
    /// The window is showing captions and nothing else.
    ///
    /// Persisted, because a caption strip you have to summon again at every launch is not the
    /// thing it is for. The risk that carries is being restored into a small window whose only
    /// exits are broken, so MainWindow applies this last, after the keyboard shortcut that
    /// leaves it has been registered.
    /// </summary>
    public bool CompactMode { get; set; }

    /// <summary>
    /// Where the window sat in each mode, kept separately.
    ///
    /// The two are genuinely different windows to a user: one is a workspace they size to
    /// their screen, the other is a strip they park at the top of a monitor over whatever
    /// they are watching. Sharing one size would drag each mode back to the other's shape
    /// every time they switched.
    ///
    /// Nullable so that "never set" stays distinguishable from a real zero, which a
    /// minimised or off-screen window can genuinely report.
    /// </summary>
    public int? CompactWidth { get; set; }
    public int? CompactHeight { get; set; }
    public int? CompactLeft { get; set; }
    public int? CompactTop { get; set; }

    public int? ExpandedWidth { get; set; }
    public int? ExpandedHeight { get; set; }
    public int? ExpandedLeft { get; set; }
    public int? ExpandedTop { get; set; }

    /// <summary>
    /// Where recordings are written. Null means the backend's own default, which is
    /// %USERPROFILE%\Sunno\Recordings.
    ///
    /// Null rather than a resolved path on purpose: writing the default in here would bake
    /// one machine's profile path into a settings file, and a folder that is only ever
    /// created when a recording is actually saved cannot be pre-empted by a default.
    /// </summary>
    public string? RecordingsPath { get; set; }

    /// <summary>
    /// Whether this machine has no settings file at all, meaning nobody has ever finished
    /// setting Sunno up here.
    ///
    /// Deliberately narrower than "these are the defaults". A file that exists but cannot be
    /// read also yields defaults, and that user is not new: dropping them onto a first-run
    /// screen would look like the app had forgotten them, which is the opposite of reassuring
    /// when their settings have just been lost. They get the ordinary window, and the
    /// unreadable file is traced instead.
    ///
    /// Used by the startup path to decide whether to open straight onto the setup screen.
    /// Without it the window shows its normal shell for as long as the Python backend takes
    /// to start, and then replaces it with the model picker, which reads as the app changing
    /// its mind about what it is.
    ///
    /// Not serialised. It describes this load, not the user's preferences.
    /// </summary>
    [JsonIgnore]
    public bool IsFirstRun { get; private set; }

    private static string FilePath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Sunno", "settings.json");

    public static AppSettings Load()
    {
        try
        {
            if (!File.Exists(FilePath))
            {
                return new AppSettings { IsFirstRun = true };
            }

            var json = File.ReadAllText(FilePath);
            var loaded = JsonSerializer.Deserialize<AppSettings>(json);
            if (loaded is not null) return loaded;

            App.Trace("settings.json deserialised to null; using defaults");
        }
        catch (Exception ex)
        {
            // A corrupt settings file must not stop the app starting. It is traced rather
            // than swallowed silently, because the failure is invisible otherwise and its
            // symptom is alarming: every preference reverts, including the chosen model,
            // which then reads as the app forgetting what it was told.
            App.Trace($"settings.json unreadable ({ex.GetType().Name}: {ex.Message}); using defaults");
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
