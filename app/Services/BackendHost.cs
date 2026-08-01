using System.Diagnostics;

namespace Sunno.Services;

/// <summary>
/// Owns the Python captioning backend as a child process.
///
/// The backend is deliberately a separate process rather than an in-proc library: it keeps a
/// ~4 GB CUDA/model working set out of the UI process, and a crash in inference leaves the
/// window alive and reconnecting rather than taking the app down.
/// </summary>
public sealed class BackendHost : IDisposable
{
    private Process? _process;
    private readonly ChildProcessJob _job = new();
    private readonly List<string> _log = new();
    private readonly object _logLock = new();

    public event Action<string>? Output;

    /// <summary>
    /// Raised when the backend exits without being asked to. Without this a crashed backend is
    /// indistinguishable from a slow-loading one: the socket simply never opens and the UI sits
    /// on "Starting the speech engine…" forever.
    /// </summary>
    public event Action<string>? Crashed;

    /// <summary>Set during Dispose so a deliberate shutdown isn't reported as a crash.</summary>
    private bool _stopping;

    public bool IsRunning => _process is { HasExited: false };

    /// <summary>Where the app is installed. Read-only when packaged as MSIX.</summary>
    private static string InstallRoot => AppContext.BaseDirectory;

    /// <summary>
    /// True when running with package identity. GetCurrentPackageFullName returns
    /// APPMODEL_ERROR_NO_PACKAGE (15700) for an unpackaged process, which is the documented
    /// way to ask — more reliable than inspecting the install path.
    /// </summary>
    public static bool IsPackaged()
    {
        try
        {
            int length = 0;
            var rc = GetCurrentPackageFullName(ref length, null);
            return rc != 15700;   // APPMODEL_ERROR_NO_PACKAGE
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Package family name, or null when unpackaged.</summary>
    private static string? PackageFamilyName()
    {
        int length = 0;
        if (GetCurrentPackageFamilyName(ref length, null) != 122) return null;   // ERROR_INSUFFICIENT_BUFFER
        var buffer = new System.Text.StringBuilder(length);
        return GetCurrentPackageFamilyName(ref length, buffer) == 0 ? buffer.ToString() : null;
    }

    [System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet =
        System.Runtime.InteropServices.CharSet.Unicode)]
    private static extern int GetCurrentPackageFamilyName(ref int length,
                                                          System.Text.StringBuilder? name);

    [System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet =
        System.Runtime.InteropServices.CharSet.Unicode)]
    private static extern int GetCurrentPackageFullName(ref int packageFullNameLength,
                                                        System.Text.StringBuilder? packageFullName);

    /// <summary>
    /// Locates the Python interpreter: the bundled runtime when packaged, otherwise the
    /// developer venv found by walking up from the build output.
    /// </summary>
    public static string? FindPython()
    {
        var bundled = Path.Combine(InstallRoot, "backend", "python", "python.exe");
        if (File.Exists(bundled)) return bundled;

        return WalkUp(dir => Path.Combine(dir, ".venv", "Scripts", "python.exe"), File.Exists);
    }

    /// <summary>Directory containing the `server` package.</summary>
    public static string? FindBackendRoot()
    {
        var bundled = Path.Combine(InstallRoot, "backend");
        if (Directory.Exists(Path.Combine(bundled, "server"))) return bundled;

        var found = WalkUp(dir => Path.Combine(dir, "server"), Directory.Exists);
        return found is null ? null : Path.GetDirectoryName(found);
    }

    /// <summary>
    /// Walks up from the build output looking for a dev-tree marker. Avoids hardcoding a
    /// relative depth, which breaks whenever the TFM or RID changes the output path.
    /// </summary>
    private static string? WalkUp(Func<string, string> candidate, Func<string, bool> exists)
    {
        var dir = new DirectoryInfo(InstallRoot);
        for (var i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
        {
            var path = candidate(dir.FullName);
            if (exists(path)) return path;
        }
        return null;
    }

    public string Start(string? device = null, string model = "large-v3", string? vocabulary = null,
                        bool startStopped = false, int? loopbackDevice = null)
    {
        if (IsRunning) return "already running";

        // Dispose latches _stopping and permanently closes the job object. Starting after a
        // Dispose would therefore produce a child that is neither crash-reported nor tied to
        // kill-on-close — a capture process able to outlive a killed UI with the microphone
        // still open. Refuse rather than start something unsafe; use Restart to cycle.
        if (_job.IsDisposed)
            return "Backend host has been shut down; restart the app.";
        _stopping = false;

        var python = FindPython();
        var root = FindBackendRoot();
        if (python is null) return "Python backend not found.";
        if (root is null) return "Backend 'server' package not found.";

        var args = new List<string> { "-m", "server.app", "--model", model };
        if (!string.IsNullOrWhiteSpace(device)) { args.Add("--device"); args.Add(device); }
        if (!string.IsNullOrWhiteSpace(vocabulary)) { args.Add("--vocabulary"); args.Add(vocabulary); }
        // Capturing an output endpoint rather than a microphone, so calls and video get
        // captioned. Mutually exclusive with --device on the backend side.
        if (loopbackDevice is int loop)
        {
            args.Add("--loopback-device");
            args.Add(loop.ToString());
        }
        // Load the model but leave the microphone alone. Used while consent is still being
        // resolved, so the ~33 s load overlaps the dialog instead of following it.
        if (startStopped) args.Add("--start-stopped");

        var psi = new ProcessStartInfo(python)
        {
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var a in args) psi.ArgumentList.Add(a);
        psi.Environment["PYTHONUTF8"] = "1";
        psi.Environment["PYTHONUNBUFFERED"] = "1";
        // Tell the backend explicitly whether it is running from a package, rather than
        // leaving paths.is_packaged() to infer it from a "WindowsApps" path substring.
        if (IsPackaged())
        {
            psi.Environment["MSIX_PACKAGE_ROOT"] = InstallRoot;
        }

        _process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _process.OutputDataReceived += (_, e) => Record(e.Data);
        _process.ErrorDataReceived += (_, e) => Record(e.Data);
        _process.Exited += (_, _) =>
        {
            if (_stopping) return;
            var code = -1;
            try { code = _process?.ExitCode ?? -1; } catch { /* raced with disposal */ }
            Crashed?.Invoke(DescribeExit(code));
        };

        try
        {
            _process.Start();
            // Tie the child to this process at the kernel level, so the microphone is
            // released even if the UI is killed rather than closed cleanly.
            if (!_job.Assign(_process))
            {
                // Never leave an unsupervised capture process behind: a backend outside the
                // job survives a killed UI still holding the microphone open. Suppress the
                // Exited handler first, or this deliberate kill would also be announced as a
                // crash on top of the error this returns.
                _stopping = true;
                try { _process.Kill(entireProcessTree: true); } catch { /* already gone */ }
                _process.Dispose();
                _process = null;
                _stopping = false;
                return "Could not supervise the speech engine; not starting it.";
            }
            _process.BeginOutputReadLine();
            _process.BeginErrorReadLine();
        }
        catch (Exception ex)
        {
            _process = null;
            return $"Failed to start backend: {ex.Message}";
        }

        return string.Empty;
    }

    /// <summary>
    /// Stop the backend and start it again on a different model.
    ///
    /// The engine is built once during startup, so switching means reloading it. Restarting the
    /// child is far less invasive than it sounds: speaker profiles are persisted server-side and
    /// the transcript lives in the UI, so only the ~30 s load is actually lost. Crash reporting
    /// is suppressed across the swap so a deliberate stop isn't announced as a failure.
    /// </summary>
    public string Restart(string? device, string model, string? vocabulary,
                          bool startStopped = false, int? loopbackDevice = null)
    {
        _stopping = true;
        try
        {
            if (_process is not null)
            {
                try
                {
                    if (!_process.HasExited)
                    {
                        _process.Kill(entireProcessTree: true);
                        _process.WaitForExit(10000);
                    }
                }
                catch { /* already gone */ }
                _process.Dispose();
                _process = null;
            }
        }
        finally
        {
            _stopping = false;
        }
        return Start(device, model, vocabulary, startStopped, loopbackDevice);
    }

    private void Record(string? line)
    {
        if (string.IsNullOrEmpty(line)) return;
        lock (_logLock)
        {
            _log.Add(line);
            if (_log.Count > 500) _log.RemoveAt(0);
            AppendToLogFile(line);
        }
        Output?.Invoke(line);
    }

    /// <summary>
    /// Mirror backend output to disk. A native crash in a bundled .pyd produces no Python
    /// traceback and no window, so without a file on disk the only evidence is a Windows Error
    /// Reporting entry the user will never think to look for.
    ///
    /// Rotated at <see cref="MaxLogBytes"/>, keeping one previous file. Before rotation existed
    /// this grew without limit for the life of the install, which on a machine used daily is a
    /// file that only ever gets bigger and is never read.
    /// </summary>
    private static void AppendToLogFile(string line)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(LogPath)!);
            RotateIfLarge();
            File.AppendAllText(LogPath, $"{DateTime.Now:HH:mm:ss} {line}{Environment.NewLine}");
        }
        catch { /* logging must never take the app down */ }
    }

    /// <summary>Size at which the log is rolled over. One previous file is kept.</summary>
    private const long MaxLogBytes = 2 * 1024 * 1024;

    private static int _sinceSizeCheck;

    private static void RotateIfLarge()
    {
        // Stat'ing the file on every line would put a syscall in the path of every line the
        // backend prints. Check on the first line of each session and periodically after that:
        // checking only every 200th would mean a user whose sessions each log fewer than 200
        // lines never checks at all, and the file grows across launches forever, which is the
        // exact condition the cap exists for.
        //
        // Interlocked because stdout and stderr are drained on two different threadpool
        // threads, so a plain ++ can lose an increment and skip the check indefinitely.
        var n = System.Threading.Interlocked.Increment(ref _sinceSizeCheck) - 1;
        if (n != 0 && n % 200 != 0) return;

        // Its own try: a failed rotation must not cost the caller its log line, which is what
        // happens if this throws into AppendToLogFile's catch. The line matters more than the
        // rollover.
        try
        {
            var info = new FileInfo(LogPath);
            if (!info.Exists || info.Length < MaxLogBytes) return;

            var previous = LogPath + ".1";
            File.Delete(previous);
            File.Move(LogPath, previous, overwrite: true);
        }
        catch { /* try again in another 200 lines */ }
    }

    public static string LogPath { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Sunno", "backend.log");

    /// <summary>
    /// Where the log actually lands. Under MSIX, writes to LocalAppData are redirected into
    /// the package's LocalCache, so showing the raw path sends the user somewhere that looks
    /// empty in Explorer.
    /// </summary>
    public static string DisplayLogPath
    {
        get
        {
            if (!IsPackaged()) return LogPath;
            var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            var packages = Path.Combine(local, "Packages");
            try
            {
                var family = PackageFamilyName();
                if (family is not null)
                    return Path.Combine(packages, family, "LocalCache", "Local", "Sunno", "backend.log");
            }
            catch { /* fall through to the unredirected path */ }
            return LogPath;
        }
    }

    /// <summary>Turn an exit code into something a non-developer can act on.</summary>
    private string DescribeExit(int code)
    {
        // Anchored on the backend's own markers, not a substring scan. The same stream carries
        // every caption, so matching "error" anywhere in a line meant a speaker saying "an
        // error" or "terror" landed in the crash banner — and could push the real traceback
        // out of the three lines shown. Putting transcript text where the failure should be is
        // worse than showing no detail at all.
        var errorLines = RecentLog()
            .Where(IsDiagnostic)
            .TakeLast(3)
            .ToList();

        // A negative or very large code means the OS terminated the process rather than it
        // exiting. The number carries nothing a user could act on, so don't show it.
        var reason = code is < 0 or > 0x40000000
            ? "The speech engine stopped unexpectedly."
            : $"The speech engine exited with code {code}.";

        return errorLines.Count == 0 ? reason : $"{reason}\n{string.Join("\n", errorLines)}";
    }

    /// <summary>
    /// Whether a log line came from the backend's diagnostics rather than from a caption.
    /// Captions are printed unprefixed, so anchoring at the start of the line separates them.
    /// Traceback frames are matched before trimming, since the indent is what identifies them.
    /// </summary>
    private static bool IsDiagnostic(string line)
    {
        if (line.StartsWith("  File \"", StringComparison.Ordinal)) return true;

        var text = line.TrimStart();
        return text.StartsWith("[fatal]", StringComparison.OrdinalIgnoreCase)
            || text.StartsWith("[error]", StringComparison.OrdinalIgnoreCase)
            || text.StartsWith("Traceback", StringComparison.Ordinal)
            // The final line of a Python traceback: "ValueError: ...", "RuntimeError: ...".
            || System.Text.RegularExpressions.Regex.IsMatch(text, @"^[A-Za-z_][\w.]*(Error|Exception|Interrupt)\s*:");
    }

    public IReadOnlyList<string> RecentLog()
    {
        lock (_logLock) return _log.ToArray();
    }

    public void Dispose()
    {
        _stopping = true;
        if (_process is not null)
        {
            try
            {
                if (!_process.HasExited)
                {
                    _process.Kill(entireProcessTree: true);
                    _process.WaitForExit(5000);
                }
            }
            catch { /* already gone */ }
            _process.Dispose();
            _process = null;
        }
        _job.Dispose();   // backstop: kills anything left in the job
    }
}
