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
    public string? Vocabulary { get; set; }
    public double CaptionFontSize { get; set; } = 26;
    public bool AlwaysOnTop { get; set; } = true;
    /// <summary>Whether the one-time microphone consent dialog has been shown.</summary>
    public bool MicrophoneAsked { get; set; }

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
            Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
            File.WriteAllText(FilePath,
                JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch
        {
            // Losing a preference is not worth surfacing an error for.
        }
    }
}
