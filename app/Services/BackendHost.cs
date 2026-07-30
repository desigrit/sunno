using System.Diagnostics;

namespace LiveCaptions.Services;

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

    public string Start(string? device = null, string model = "large-v3", string? vocabulary = null)
    {
        if (IsRunning) return "already running";

        var python = FindPython();
        var root = FindBackendRoot();
        if (python is null) return "Python backend not found.";
        if (root is null) return "Backend 'server' package not found.";

        var args = new List<string> { "-m", "server.app", "--model", model };
        if (!string.IsNullOrWhiteSpace(device)) { args.Add("--device"); args.Add(device); }
        if (!string.IsNullOrWhiteSpace(vocabulary)) { args.Add("--vocabulary"); args.Add(vocabulary); }

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

    private void Record(string? line)
    {
        if (string.IsNullOrEmpty(line)) return;
        lock (_logLock)
        {
            _log.Add(line);
            if (_log.Count > 500) _log.RemoveAt(0);
        }
        Output?.Invoke(line);
    }

    public IReadOnlyList<string> RecentLog()
    {
        lock (_logLock) return _log.ToArray();
    }

    public void Dispose()
    {
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
