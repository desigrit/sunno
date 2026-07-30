using Microsoft.UI.Xaml;

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
            File.AppendAllText(CrashLog,
                $"{DateTime.Now:O}\n{ex.GetType().FullName}: {ex.Message}\n{ex.StackTrace}\n\n");
        }
        catch { /* nothing useful to do if even logging fails */ }
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        try
        {
            _window = new MainWindow();
            _window.Activate();
        }
        catch (Exception ex)
        {
            Log(ex);
            throw;
        }
    }
}
