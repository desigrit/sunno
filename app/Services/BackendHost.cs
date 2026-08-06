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
    private readonly HashSet<Process> _expectedExits = [];

    /// <summary>
    /// Whether the engine reported that it could not be loaded at all. Distinct from "no GPU":
    /// the CPU path is dead too, so the process is going to die whatever the user does next.
    /// Guarded by <see cref="_logLock"/>.
    /// </summary>
    private bool _engineUnloadable;

    public event Action<string>? Output;

    /// <summary>
    /// Raised when the backend exits without being asked to. Without this a crashed backend is
    /// indistinguishable from a slow-loading one: the socket simply never opens and the UI sits
    /// on "Starting the speech engine…" forever.
    /// </summary>
    /// <summary>Fired when the backend exits unexpectedly: a human reason, and the detail.</summary>
    public event Action<string, string>? Crashed;

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
                        bool startStopped = false, string? loopbackDevice = null,
                        string? computeDevice = null)
    {
        if (IsRunning) return "already running";

        // Dispose permanently closes the job object. Starting after a
        // Dispose would therefore produce a child that is neither crash-reported nor tied to
        // kill-on-close — a capture process able to outlive a killed UI with the microphone
        // still open. Refuse rather than start something unsafe; use Restart to cycle.
        if (_job.IsDisposed)
            return "Sunno needs to be restarted before speech recognition can start again.";
        // Cleared per launch so a repaired or transient load failure does not poison a restart.
        lock (_logLock)
        {
            _engineUnloadable = false;
        }

        var python = FindPython();
        var root = FindBackendRoot();
        // Both mean the install is incomplete, which is one situation to the person looking at
        // the banner even though it is two causes here. The remedy is identical - reinstall - so
        // the distinction is no use to them, but it is the first thing a maintainer wants from a
        // bug report. It goes to App.Trace rather than Record because the diagnostics export
        // reads startup-trace.log and deliberately does not read backend.log.
        if (python is null)
        {
            App.Trace("engine not started: no Python runtime found");
            return "Part of Sunno is missing, so speech recognition cannot start. "
                   + "Reinstalling Sunno should replace it.";
        }
        if (root is null)
        {
            App.Trace("engine not started: 'server' package not found");
            return "Part of Sunno is missing, so speech recognition cannot start. "
                   + "Reinstalling Sunno should replace it.";
        }

        var args = new List<string> { "-m", "server.app", "--model", model };
        if (!string.IsNullOrWhiteSpace(device)) { args.Add("--device"); args.Add(device); }
        if (!string.IsNullOrWhiteSpace(vocabulary)) { args.Add("--vocabulary"); args.Add(vocabulary); }
        // Capturing an output endpoint rather than a microphone, so calls and video get
        // captioned. Mutually exclusive with --device on the backend side.
        if (!string.IsNullOrWhiteSpace(loopbackDevice))
        {
            args.Add("--loopback-device");
            args.Add(loopbackDevice);
        }
        // Load the model but leave the microphone alone. Used while consent is still being
        // resolved, so the ~33 s load overlaps the dialog instead of following it.
        if (startStopped) args.Add("--start-stopped");
        // Omitted rather than passed as "auto" when the user has expressed no preference, so
        // the backend's own default stays the single definition of what "auto" means.
        if (!string.IsNullOrEmpty(computeDevice) && computeDevice != "auto")
        {
            args.Add("--compute-device");
            args.Add(computeDevice);
        }

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

        var outputClosed = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var errorClosed = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _process = process;
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data is null) outputClosed.TrySetResult();
            else Record(e.Data);
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is null) errorClosed.TrySetResult();
            else Record(e.Data);
        };
        process.Exited += async (_, _) =>
        {
            // Exited can run before the final redirected line is delivered. Wait for both EOF
            // notifications so a fast startup failure cannot outrun its actionable marker.
            await Task.WhenAll(outputClosed.Task, errorClosed.Task).ConfigureAwait(false);
            if (ConsumeExpectedExit(process)) return;
            var code = -1;
            try { code = process.ExitCode; } catch { /* raced with disposal */ }
            var (reason, detail) = DescribeExit(code);
            Crashed?.Invoke(reason, detail);
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
                ExpectExit(process);
                // Readers have not started, so no EOF callbacks will arrive for this process.
                outputClosed.TrySetResult();
                errorClosed.TrySetResult();
                try { _process.Kill(entireProcessTree: true); } catch { /* already gone */ }
                _process.Dispose();
                _process = null;
                return "Could not supervise the speech engine; not starting it.";
            }
            _process.BeginOutputReadLine();
            _process.BeginErrorReadLine();
        }
        catch (Exception ex)
        {
            _process = null;
            // Same split as DescribeExit: a friendly sentence on screen, the exception text kept
            // for whoever reads the bug report. It goes to App.Trace, not Record, because this
            // failure means no process ever started - nothing will call DescribeExit, so the
            // export's crash section stays empty and startup-trace.log is the only part of the
            // export that can carry it. This used to return ex.Message directly, which put raw
            // framework text - and for a missing file, an absolute path - on the banner as the
            // whole explanation.
            App.Trace($"engine failed to start: {ex.GetType().Name}: {ex.Message}");
            return "Sunno could not start its speech engine. Restarting Sunno usually clears "
                   + "this; if it keeps happening, reinstalling will replace a missing file.";
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
                          bool startStopped = false, string? loopbackDevice = null,
                          string? computeDevice = null)
    {
        if (_process is not null)
        {
            var process = _process;
            ExpectExit(process);
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                    if (!process.WaitForExit(10000))
                    {
                        App.Trace("engine restart aborted: old process did not exit after kill");
                        return "Sunno could not stop the previous speech engine. Restart the app "
                               + "before changing models.";
                    }
                }
                // The timed overload only confirms process termination. The parameterless call
                // additionally waits for both asynchronous output readers to finish, preventing
                // old diagnostics from racing the replacement launch's state reset.
                process.WaitForExit();
            }
            catch (InvalidOperationException)
            {
                // It exited between HasExited and Kill; redirected output still has to drain.
                try
                {
                    process.WaitForExit();
                }
                catch (Exception ex)
                {
                    App.Trace($"engine restart drain failed: {ex.GetType().Name}: {ex.Message}");
                    return "Sunno could not stop the previous speech engine. Restart the app "
                           + "before changing models.";
                }
            }
            catch (Exception ex)
            {
                App.Trace($"engine restart aborted: {ex.GetType().Name}: {ex.Message}");
                return "Sunno could not stop the previous speech engine. Restart the app before "
                       + "changing models.";
            }
            process.Dispose();
            _process = null;
        }
        return Start(device, model, vocabulary, startStopped, loopbackDevice, computeDevice);
    }

    private void ExpectExit(Process process)
    {
        lock (_logLock) _expectedExits.Add(process);
    }

    private bool ConsumeExpectedExit(Process process)
    {
        lock (_logLock) return _expectedExits.Remove(process);
    }

    private void Record(string? line)
    {
        if (string.IsNullOrEmpty(line)) return;
        lock (_logLock)
        {
            // Latched here rather than searched for at exit time. The engine announces this
            // failure early in startup, and the traceback that follows when the model finally
            // loads would push it out of the handful of lines RecentDiagnostics keeps. Only the
            // boolean is retained, never the line, so this cannot become a second route
            // for backend text to reach the export.
            if (line.Contains("speech engine could not be loaded", StringComparison.Ordinal))
                _engineUnloadable = true;

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

    /// <summary>
    /// Turn an exit code into something a non-developer can act on, plus the technical detail
    /// separately.
    ///
    /// Split deliberately. The banner used to carry both, so a port conflict presented the user
    /// with a Python traceback full of absolute paths as the entire explanation. The reason goes
    /// on screen; the detail goes behind "Copy details", where someone filing a bug report can
    /// still get all of it.
    /// </summary>
    private (string Reason, string Detail) DescribeExit(int code)
    {
        // Anchored on the backend's own markers, not a substring scan. The same stream carries
        // every caption, so matching "error" anywhere in a line meant a speaker saying "an
        // error" or "terror" landed in the crash banner — and could push the real traceback
        // out of the three lines shown. Putting transcript text where the failure should be is
        // worse than showing no detail at all.
        var detail = RecentDiagnostics();

        bool unloadable;
        lock (_logLock)
        {
            unloadable = _engineUnloadable;
        }

        // Name the cause when the backend's own output identifies it. Anything not recognised
        // falls through to the generic wording rather than guessing: a wrong explanation sends
        // someone hunting for a problem they do not have.
        //
        // Order is deliberate, not incidental: the two "could not load" branches sit above the
        // exit-code test and shadow it, so a latched load failure reports its cause rather than
        // "stopped unexpectedly". A specific cause should always beat a generic one, and the OS
        // termination code carries nothing extra once the reason is already known.
        //
        // Deliberately not mapped: PortAudio errno values. -9998 is paInvalidChannelCount and
        // appears on nearly every failed open because _candidate_formats probes several channel
        // counts, so keying off it would shadow the real cause. Microphone failures already
        // arrive as human sentences from server/audio.py, with more information than an errno
        // has, and go through ShowActionableError instead.
        string reason;
        if (detail.Contains("10048") || detail.Contains("only one usage of each socket"))
        {
            reason = "Sunno's speech engine could not start because its connection was still in use. "
                     + "Restarting Sunno should clear it.";
        }
        else if (unloadable)
        {
            reason = "Sunno's speech engine could not load on this PC. Reinstalling Sunno usually "
                     + "repairs it.";
        }
        else if (code is < 0 or > 0x40000000)
        {
            // A negative or very large code means the OS terminated the process rather than
            // it exiting. The number carries nothing a user could act on, so don't show it.
            reason = "Sunno's speech engine stopped unexpectedly.";
        }
        else
        {
            reason = "Sunno's speech engine stopped.";
        }

        return (reason, detail);
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

    /// <summary>
    /// The last few lines the engine printed about a problem, narrowed to its own diagnostics.
    ///
    /// The only sanctioned way to put the engine's words in front of a user or into the
    /// diagnostics export. RecentLog() is the raw stream and carries every caption; this
    /// applies IsDiagnostic, which anchors on the backend's own markers so transcript text is
    /// excluded by construction rather than by hoping the message looks technical.
    ///
    /// A caller that reaches past this and uses a backend-supplied string directly is how the
    /// export's promise - no transcript text, no speaker names, no device names - stops being
    /// enforced by anything.
    /// </summary>
    public string RecentDiagnostics(int lines = 3) =>
        string.Join("\n", RecentLog().Where(IsDiagnostic).TakeLast(lines));

    public void Dispose()
    {
        if (_process is not null)
        {
            ExpectExit(_process);
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
