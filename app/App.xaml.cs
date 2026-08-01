using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;
using Microsoft.Windows.AppLifecycle;

namespace Sunno;

public partial class App : Application
{
    private Window? _window;

    /// <summary>Startup/XAML failures otherwise surface only as an opaque 0xc000027b crash.</summary>
    private static readonly string CrashLog = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Sunno", "startup-error.log");

    public App()
    {
        InitializeComponent();
        UnhandledException += (_, e) =>
        {
            Log(e.Exception);
            e.Handled = true;
        };
        AppDomain.CurrentDomain.UnhandledException += (_, e) => Log(e.ExceptionObject as Exception);
        TaskScheduler.UnobservedTaskException += (_, e) => Log(e.Exception);
    }

    private static void Log(Exception? ex)
    {
        if (ex is null) return;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(CrashLog)!);

            // Bounded, unlike before. A crash loop can fire this handler repeatedly, and the
            // file was appended to for the life of the install with nothing ever trimming it.
            // Truncating on overflow rather than rotating: the most recent crashes are the ones
            // that explain the state the user is in now, and a second file to reason about is
            // not worth it for a log this small.
            const long maxBytes = 256 * 1024;
            var info = new FileInfo(CrashLog);
            if (info.Exists && info.Length > maxBytes)
                File.WriteAllText(CrashLog,
                    $"{DateTime.Now:O}\n(earlier entries dropped; the log exceeded {maxBytes / 1024} KB)\n\n");

            File.AppendAllText(CrashLog,
                $"{DateTime.Now:O}\n{ex.GetType().FullName}: {ex.Message}\n{ex.StackTrace}\n\n");
        }
        catch { /* nothing useful to do if even logging fails */ }
    }

    /// <summary>
    /// Startup breadcrumbs, rewritten each launch.
    ///
    /// A XAML failure on the UI thread becomes a stowed exception (0xC000027B) that kills the
    /// process without reaching UnhandledException, so the only way to find out how far
    /// startup got is to record it as it happens. This is what located the repeated
    /// AppCapability.CheckAccess call that made the packaged app unlaunchable.
    /// </summary>
    public static void Trace(string stage)
    {
        try
        {
            var dir = Path.GetDirectoryName(CrashLog)!;
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, "startup-trace.log");
            // Truncate on the first write of each launch so the file stays a single startup.
            if (!_traceStarted)
            {
                _traceStarted = true;
                File.WriteAllText(path, $"--- {DateTime.Now:O} ---{Environment.NewLine}");
            }
            File.AppendAllText(path, $"{DateTime.Now:HH:mm:ss.fff}  {stage}{Environment.NewLine}");
        }
        catch { /* diagnostics must never break startup */ }
    }

    private static bool _traceStarted;

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        try
        {
            Trace("OnLaunched");
            // One backend per machine: a second instance would spawn a second Python child that
            // cannot bind port 8766, so it dies and leaves a dead-looking window behind. Hand the
            // activation to the running instance and get out of the way instead.
            var primary = AppInstance.FindOrRegisterForKey("Sunno");
            if (!primary.IsCurrent)
            {
                Trace("redirecting to primary");
                RedirectAndExit(primary);
                return;
            }
            primary.Activated += (_, _) => _window?.DispatcherQueue.TryEnqueue(BringToFront);

            Trace("creating MainWindow");
            _window = new MainWindow();
            Trace("activating MainWindow");
            _window.Activate();
            Trace("activated");
        }
        catch (Exception ex)
        {
            Trace($"OnLaunched THREW: {ex.GetType().Name}: {ex.Message}");
            Log(ex);
            throw;
        }
    }

    /// <summary>
    /// Forward this launch to the instance that owns the backend, then quit.
    ///
    /// RedirectActivationToAsync must not be awaited on the UI thread, so it runs on the pool
    /// and this waits with a timeout: failing to redirect only costs the window not coming
    /// forward, whereas hanging here would leave an invisible zombie process holding a job
    /// object. Exiting is the important half.
    /// </summary>
    private static void RedirectAndExit(AppInstance primary)
    {
        var done = new ManualResetEventSlim(false);
        _ = Task.Run(async () =>
        {
            try { await primary.RedirectActivationToAsync(AppInstance.GetCurrent().GetActivatedEventArgs()); }
            catch (Exception ex) { Log(ex); }
            finally { done.Set(); }
        });
        done.Wait(TimeSpan.FromSeconds(5));
        Trace("RedirectAndExit -> Environment.Exit");
        Environment.Exit(0);
    }

    /// <summary>Activate() alone doesn't reliably raise a background window, so ask Win32.</summary>
    private void BringToFront()
    {
        if (_window is null) return;
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(_window);
        if (IsIconic(hwnd)) ShowWindow(hwnd, 9);   // SW_RESTORE
        SetForegroundWindow(hwnd);
    }

    [DllImport("user32.dll")] private static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] private static extern bool IsIconic(IntPtr hWnd);
}
