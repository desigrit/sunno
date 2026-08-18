using System.Text;
using Windows.ApplicationModel;

namespace Sunno.Services;

/// <summary>
/// Builds the report a user attaches to a bug report.
///
/// Written as an allow-list, not a filter, and that distinction is the entire design. Sunno's
/// claim is that conversations never leave the machine, so the one feature whose purpose is to
/// send a file to a stranger is the one place that claim is easiest to break. A filter has to
/// anticipate every category of secret; an allow-list only emits what someone deliberately put
/// on it.
///
/// Deliberately NOT included, each for a specific reason:
///
///   backend.log        Append-only, never truncated before rotation was added, and until very
///                      recently it recorded every finalised caption. On any machine that ran an
///                      earlier build it still holds verbatim conversation, including other
///                      people who never agreed to anything.
///   speakers.json      Pinned speakers: a person's name next to a voice fingerprint.
///   Vocabulary         Free text to bias recognition. No screen currently offers a way
///                      to set it, so in practice it is
///                      colleague names, project names and places.
///   Device names       "Headset (R-Phonak hearing aid)" tells the reader the user wears a
///                      hearing aid. That is health information, disclosed by a field nobody
///                      would think of as sensitive. Only whether a device is chosen is
///                      reported, never which.
///   Transcript text    Never held here at all, and never should be.
/// </summary>
public static class Diagnostics
{
    public static string Build(AppSettings settings, string? activeModel, string? computeDevice,
                               bool backendRunning, bool connected)
    {
        var sb = new StringBuilder();
        sb.AppendLine("Sunno diagnostics");
        sb.AppendLine($"Generated       {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine();

        sb.AppendLine("-- Build --");
        sb.AppendLine($"App version     {AppVersion()}");
        sb.AppendLine($"Package         {PackageIdentity()}");
        // Both architectures, because they differ on Windows-on-ARM and only the pair is
        // meaningful. An x64 build runs there under emulation and reports ProcessArchitecture
        // X64, so every report from an ARM machine was indistinguishable from one sent by an
        // Intel machine - which is exactly the case most likely to be generating the report.
        sb.AppendLine($"Architecture    {ArchitectureLine()}");
        sb.AppendLine($".NET            {Environment.Version}");
        sb.AppendLine();

        sb.AppendLine("-- System --");
        sb.AppendLine($"Windows         {Environment.OSVersion.Version}");
        sb.AppendLine($"Processors      {Environment.ProcessorCount}");
        sb.AppendLine();

        sb.AppendLine("-- Engine --");
        // The compute device comes from the model catalogue frame, where the backend sends
        // settings.device, and never from the status frame, whose "device" field is the *audio*
        // device name. Reading that one printed a hearing aid's name into a report that promised
        // it held no device names. hardware.json is not a substitute either: its keys only name
        // a compute device after five utterances have been timed, so it is empty on exactly the
        // fresh installs and startup failures that produce bug reports.
        sb.AppendLine($"Compute device  {computeDevice ?? "unknown"}");
        sb.AppendLine($"Model in use    {activeModel ?? "unknown"}");
        sb.AppendLine($"Model setting   {settings.Model}");
        sb.AppendLine($"Backend process {(backendRunning ? "running" : "not running")}");
        sb.AppendLine($"WebSocket       {(connected ? "connected" : "not connected")}");
        sb.AppendLine();

        sb.AppendLine("-- Capture --");
        // Whether, never which. See the class comment.
        var loopback = settings.LoopbackDeviceIndex is not null;
        sb.AppendLine($"Source          {(loopback ? "system audio" : "microphone")}");
        sb.AppendLine($"Device chosen   {(HasDevice(settings) ? "yes" : "no, using system default")}");
        sb.AppendLine($"Device name set {(HasDeviceName(settings) ? "yes" : "no")}");
        sb.AppendLine();

        sb.AppendLine("-- Preferences --");
        sb.AppendLine($"Caption size    {settings.CaptionFontSize}");
        sb.AppendLine($"Always on top   {settings.AlwaysOnTop}");
        sb.AppendLine($"Vocabulary set  {(string.IsNullOrWhiteSpace(settings.Vocabulary) ? "no" : "yes")}");
        sb.AppendLine();

        sb.AppendLine("-- Measured performance --");
        sb.AppendLine(HardwareJson());

        return sb.ToString();
    }

    /// <summary>
    /// The full text a user saves and sends when asked for "the logs".
    ///
    /// Deliberately does NOT include backend.log.
    ///
    /// That file is the backend's raw stdout, and raw stdout is not safe to forward. Until
    /// 42b44f5 it recorded every finalised caption, so on any machine that ran an earlier build
    /// it still holds verbatim conversation — on the machine this was written on, 316 lines of
    /// it including a work meeting. The transcript print is also still reachable today behind
    /// --echo-transcript, so this is not purely historical. Filtering it was designed and
    /// reviewed three times; each round found a way the filter let something through or stripped
    /// everything useful, and the filter would have been the only thing between a user and a
    /// disclosure.
    ///
    /// What is included is safe by construction rather than by filtering:
    ///   * the diagnostics report, an allow-list of named fields
    ///   * startup-trace.log, which records stage names, indices and outcomes only
    ///   * startup-error.log, which is .NET exception text written by App.Log
    ///   * the last crash detail, already narrowed by BackendHost.IsDiagnostic
    ///
    /// A user who genuinely needs to send backend.log can still do it deliberately: the crash
    /// banner names the path. That is a decision they make with their eyes open, which is not
    /// the same as an app bundling it for them.
    /// </summary>
    public static string BuildExport(AppSettings settings, string? activeModel,
                                     string? computeDevice, bool backendRunning, bool connected,
                                     string? lastCrashDetail)
    {
        var sb = new StringBuilder();
        sb.AppendLine("================ Sunno diagnostics ================");
        sb.AppendLine();
        sb.AppendLine("Includes: this report, startup breadcrumbs, the .NET error log and the");
        sb.AppendLine("last engine failure. Excludes transcript text, speaker names, voice");
        sb.AppendLine("fingerprints, vocabulary, device names and the speech engine's log.");
        sb.AppendLine("File paths below may contain your Windows user name.");
        sb.AppendLine();
        sb.AppendLine(Build(settings, activeModel, computeDevice, backendRunning, connected));

        sb.AppendLine();
        sb.AppendLine("================ Last engine failure ================");
        sb.AppendLine(string.IsNullOrWhiteSpace(lastCrashDetail)
            ? "(none this session)"
            : lastCrashDetail);

        sb.AppendLine();
        sb.AppendLine("================ Startup breadcrumbs ================");
        sb.AppendLine(ReadLocal("startup-trace.log"));

        sb.AppendLine();
        sb.AppendLine("================ Errors ================");
        sb.AppendLine(ReadLocal("startup-error.log"));

        return sb.ToString();
    }

    private static string ReadLocal(string name)
    {
        try
        {
            var path = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Sunno", name);
            if (!File.Exists(path)) return "(none)";
            var text = File.ReadAllText(path).Trim();
            return text.Length == 0 ? "(empty)" : text;
        }
        catch (Exception ex)
        {
            return $"(unreadable: {ex.GetType().Name})";
        }
    }

    private static bool HasDevice(AppSettings s) =>        s.DeviceIndex is not null || s.LoopbackDeviceIndex is not null;

    private static bool HasDeviceName(AppSettings s) =>
        !string.IsNullOrEmpty(s.DeviceName) || !string.IsNullOrEmpty(s.LoopbackDeviceName);

    internal static string AppVersion()
    {
        try
        {
            var v = Package.Current.Id.Version;
            return $"{v.Major}.{v.Minor}.{v.Build}.{v.Revision}";
        }
        catch
        {
            // Unpackaged: no package identity, so fall back to the assembly.
            return typeof(Diagnostics).Assembly.GetName().Version?.ToString() ?? "unknown";
        }
    }

    private static string ArchitectureLine()
    {
        var process = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture;
        var os = System.Runtime.InteropServices.RuntimeInformation.OSArchitecture;
        // Only worth spelling out when they disagree. On a matched machine the second half
        // would be noise in a report someone is meant to read before sending.
        return process == os ? $"{process}" : $"{process} process on {os} OS (emulated)";
    }

    private static string PackageIdentity()
    {
        try { return Package.Current.Id.FamilyName; }
        catch { return "unpackaged"; }
    }

    /// <summary>
    /// The learned decode timings. Numbers about this machine's speed, with nothing in them
    /// about the person using it, which is why this is the one file copied wholesale.
    /// </summary>
    private static string HardwareJson()
    {
        try
        {
            var path = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Sunno", "hardware.json");
            return File.Exists(path) ? File.ReadAllText(path).Trim() : "(not measured yet)";
        }
        catch
        {
            return "(unreadable)";
        }
    }
}
