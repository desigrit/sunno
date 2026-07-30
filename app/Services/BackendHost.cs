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
            _job.Assign(_process);
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
    /// </summary>
    private static void AppendToLogFile(string line)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(LogPath)!);
            File.AppendAllText(LogPath, $"{DateTime.Now:HH:mm:ss} {line}{Environment.NewLine}");
        }
        catch { /* logging must never take the app down */ }
    }

    public static string LogPath { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Sunno", "backend.log");

    /// <summary>Turn an exit code into something a non-developer can act on.</summary>
    private string DescribeExit(int code)
    {
        // 0xC0000005 and friends arrive as a negative exit code: the backend was terminated by
        // the OS rather than exiting, which means a native fault in a bundled module.
        var native = code is < 0 or > 0x40000000;
        var tail = string.Join(Environment.NewLine, RecentLog().TakeLast(6));

        var reason = native
            ? $"The speech engine stopped unexpectedly (0x{code:X8})."
            : $"The speech engine exited with code {code}.";

        return string.IsNullOrWhiteSpace(tail) ? reason : $"{reason}\n{tail}";
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
