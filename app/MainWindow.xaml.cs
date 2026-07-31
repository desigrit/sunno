using System;
using System.Collections.ObjectModel;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Sunno.Models;
using Sunno.Services;
using Microsoft.UI.Composition.SystemBackdrops;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Animation;
using Windows.Graphics;
using Windows.Security.Authorization.AppCapabilityAccess;

namespace Sunno;

/// <summary>A microphone the backend can capture from.</summary>
public sealed record AudioDevice(int Index, string Name, string HostApi, bool Loopback = false);

/// <summary>A model shown in first-run setup.</summary>
public sealed record ModelChoice(string Id, string Name, string Detail, int ApproxMb, bool Available,
                                 string LagText = "", bool Responsive = true)
{
    public string SizeLabel => ApproxMb >= 1024
        ? $"{ApproxMb / 1024.0:0.0} GB"
        : $"{ApproxMb} MB";

    /// <summary>
    /// Speed line for the first-run picker. This is the one screen where the number really
    /// matters: the choice made here costs a multi-gigabyte download, and on a CPU-only
    /// machine the most accurate model runs several seconds behind — which is fine for
    /// captioning a video and useless for following a conversation.
    /// </summary>
    public string SpeedLabel => Responsive
        ? $"Captions {LagText}"
        : $"Captions {LagText} — fine for video, too slow for conversation";

    public Visibility SpeedVisibility =>
        string.IsNullOrEmpty(LagText) ? Visibility.Collapsed : Visibility.Visible;
}

public sealed partial class MainWindow : Window
{
    private const int MaxLines = 200;
    private const double MinFont = 16, MaxFont = 56;

    private readonly CaptionClient _client = new();
    private readonly BackendHost _backend = new();
    private readonly DispatcherQueue _ui;
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(5) };
    private readonly AppSettings _settings = AppSettings.Load();

    private CaptionLine? _provisional;
    private int _currentUtterance = -1;
    private bool _running = true;
    private bool _suppressDeviceEvent;
    private bool _backendLoading = true;
    /// <summary>Set while a microphone failure is unresolved, so transient status updates
    /// cannot erase the explanation before the user has read it.</summary>
    private bool _micProblem;
    /// <summary>WebSocket reachability, mirrored so capture can be deferred until it's up.</summary>
    private bool _connected;
    /// <summary>Microphone consent is settled in our favour.</summary>
    private bool _micGranted;
    /// <summary>The backend was launched with --start-stopped and still needs a start command.</summary>
    private bool _startedPaused;
    /// <summary>Guards against sending start twice when consent and the socket both settle.</summary>
    private bool _captureRequested;
    private bool _micPromptDone;
    /// <summary>
    /// The model we know actually loads. A switch is only committed once the new backend comes
    /// back up, so a model that fails to load can never become the persisted choice.
    /// </summary>
    private string _lastGoodModel = string.Empty;
    /// <summary>
    /// True only between restarting the backend and it reconnecting. Distinguishes a real
    /// post-restart reconnect from an incidental socket drop during a long download, which
    /// would otherwise be mistaken for the switch completing.
    /// </summary>
    private bool _awaitingSwitchReconnect;
    /// <summary>Stops a failed fallback from bouncing between models forever.</summary>
    private bool _recoveringModel;
    /// <summary>Keeps an explanatory notice up until the user dismisses it themselves.</summary>
    private bool _infoSticky;
    /// <summary>
    /// The engine has reported ready at least once since the current backend started. Separates
    /// "this model never loads" from "something broke after it had been working".
    /// </summary>
    private bool _engineReadyThisSession;
    /// <summary>Consent as read once at startup; re-reading it later proved fatal.</summary>
    private AppCapabilityAccessStatus? _micStatus;
    /// <summary>Whether the InfoBar's action can still raise the dialog, or must fall back to
    /// Settings because Windows will not prompt a second time.</summary>
    private bool _micCanPrompt;
    /// <summary>The backend died; stop reporting progress that will never happen.</summary>
    private bool _backendFatal;
    /// <summary>Set while the engine is reloading onto a different model.</summary>
    private string? _switchingTo;
    /// <summary>Suppresses the Checked handler while the list is rebuilt from the backend.</summary>
    private bool _suppressModelEvent;

    /// <summary>
    /// How long the microphone has been open for this capture run. A Stopwatch rather than a
    /// wall-clock start time so an NTP correction or a daylight-saving jump can't make the
    /// counter leap or run backwards.
    /// </summary>
    private readonly System.Diagnostics.Stopwatch _captureClock = new();
    private readonly DispatcherTimer _elapsedTimer = new() { Interval = TimeSpan.FromSeconds(1) };
    /// <summary>
    /// Time since the backend last reported an audio level. The backend publishes one roughly
    /// every 100 ms while capturing, silence included, so a long gap means audio has stopped
    /// arriving even though the process is alive.
    /// </summary>
    private readonly System.Diagnostics.Stopwatch _sinceLevel = new();
    /// <summary>How long without a level report before the clock stops claiming all is well.</summary>
    private static readonly TimeSpan AudioStallAfter = TimeSpan.FromSeconds(4);

    /// <summary>Caption text size; the item templates read this.</summary>
    public static double CaptionSize { get; private set; } = 26;

    public ObservableCollection<CaptionLine> Lines { get; } = new();
    public ObservableCollection<SpeakerRow> Speakers { get; } = new();
    public ObservableCollection<ModelRow> Models { get; } = new();

    public MainWindow()
    {
        App.Trace("MainWindow ctor: InitializeComponent");
        InitializeComponent();
        App.Trace("MainWindow ctor: XAML loaded");
        _ui = DispatcherQueue.GetForCurrentThread();

        ConfigureWindow();
        App.Trace("MainWindow ctor: window configured");

        _client.Partial += ev => _ui.TryEnqueue(() => OnPartial(ev));
        _client.Final += ev => _ui.TryEnqueue(() => OnFinal(ev));
        _client.Discarded += id => _ui.TryEnqueue(() => OnDiscarded(id));
        _client.Level += lv => _ui.TryEnqueue(() => OnLevel(lv));
        _client.Status += st => _ui.TryEnqueue(() => OnStatus(st));
        _client.Roster += r => _ui.TryEnqueue(() => OnRoster(r));
        _client.ConnectionChanged += ok => _ui.TryEnqueue(() => OnConnection(ok));
        _client.ModelRequired += m => _ui.TryEnqueue(() => OnModelRequired(m));
        _client.DownloadProgress += p => _ui.TryEnqueue(() => OnDownloadProgress(p));
        _client.DownloadComplete += _ => _ui.TryEnqueue(OnDownloadComplete);
        _client.DownloadFailed += msg => _ui.TryEnqueue(() => OnDownloadFailed(msg));
        _client.ModelCatalog += (current, list) => _ui.TryEnqueue(() => OnModelCatalog(current, list));
        _backend.Crashed += msg => _ui.TryEnqueue(() => OnBackendCrashed(msg));

        Closed += (_, _) =>
        {
            App.Trace("MainWindow Closed -> exiting");
            MicrophoneAccess.Changed -= OnMicAccessChanged;
            _elapsedTimer.Stop();
            _ = _client.DisposeAsync();
            _backend.Dispose();
            // WinUI doesn't end the process when the last window closes; without this the app
            // lingers invisibly (and, before the job object, kept the microphone open).
            Application.Current.Exit();
        };

        _elapsedTimer.Tick += (_, _) => TickElapsed();

        // Consent has to be settled before anything opens the microphone. On first run we haven't
        // asked yet, so the backend starts paused regardless of what Windows reports — otherwise
        // the microphone would already be live behind our own consent dialog, which would make
        // asking dishonest.
        var micStatus = MicrophoneAccess.Check();
        _micStatus = micStatus;
        _lastGoodModel = _settings.Model;
        _micGranted = (micStatus is null or AppCapabilityAccessStatus.Allowed)
                      && _settings.MicrophoneAsked;
        _startedPaused = !_micGranted;

        var error = _backend.Start(
            device: _settings.DeviceIndex?.ToString(),
            model: _settings.Model,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused,
            loopbackDevice: _settings.LoopbackDeviceIndex);
        if (!string.IsNullOrEmpty(error)) SetStatus(error);
        App.Trace($"backend.Start -> {(string.IsNullOrEmpty(error) ? "ok" : error)}");
        _client.Start();
        _ = LoadDevicesAsync();
        App.Trace("MainWindow ctor: backend started");

        // Consent is asked from the content's Loaded event rather than window activation: a
        // window that is shown without being focused still needs to ask, otherwise the backend
        // sits paused behind a dialog that never appears.
        if (Content is FrameworkElement root)
        {
            root.Loaded += (_, _) =>
            {
                App.Trace("content Loaded");
                if (_micPromptDone) return;
                _micPromptDone = true;
                _ = EnsureMicrophoneAccessAsync();
            };
        }
    }

    /// <summary>
    /// Load the title-bar icon straight off disk.
    ///
    /// The unplated variant, which is what the taskbar draws: the plated one carries a solid
    /// accent backplate that looks wrong against Mica in a title bar.
    ///
    /// Not ms-appx:/// — that resolves through the package resource index, and the packaging
    /// script copies Assets in after publish, so they are present as files but absent from
    /// resources.pri. Reading the file directly works identically packaged and unpackaged.
    /// </summary>
    private void ApplyTitleBarIcon()
    {
        try
        {
            var path = Path.Combine(AppContext.BaseDirectory, "Assets",
                                    "Square44x44Logo.targetsize-32_altform-unplated.png");
            if (File.Exists(path))
                TitleBarIcon.Source = new Microsoft.UI.Xaml.Media.Imaging.BitmapImage(new Uri(path));
        }
        catch (Exception ex)
        {
            // A missing icon is cosmetic; the title bar still reads correctly without it.
            System.Diagnostics.Debug.WriteLine($"title bar icon skipped: {ex.Message}");
        }
    }

    /// <summary>Mica, extended title bar and a medium default size, matching inbox apps.</summary>
    private void ConfigureWindow()
    {
        Title = "Sunno";
        ApplyTitleBarIcon();

        if (MicaController.IsSupported())
            SystemBackdrop = new MicaBackdrop { Kind = MicaKind.Base };

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);

        AppWindow.Resize(new SizeInt32(1040, 660));
        if (AppWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.IsAlwaysOnTop = true;   // captions need to stay readable over other apps
            presenter.PreferredMinimumWidth = 720;
            presenter.PreferredMinimumHeight = 420;
        }
    }

    // ---------- caption stream ----------

    private void OnPartial(CaptionEvent ev)
    {
        if (string.IsNullOrWhiteSpace(ev.Text)) return;
        EmptyState.Visibility = Visibility.Collapsed;

        if (_provisional is null || _currentUtterance != ev.Id)
        {
            _provisional = new CaptionLine { UtteranceId = ev.Id };
            _currentUtterance = ev.Id;
            Lines.Add(_provisional);
            Trim();
        }

        Apply(_provisional, ev, isFinal: false);
        ScrollToEnd();
    }

    private void OnFinal(CaptionEvent ev)
    {
        if (string.IsNullOrWhiteSpace(ev.Text))
        {
            OnDiscarded(ev.Id);
            return;
        }

        EmptyState.Visibility = Visibility.Collapsed;
        var line = _provisional is not null && _currentUtterance == ev.Id
            ? _provisional
            : AddLine(ev.Id);

        Apply(line, ev, isFinal: true);
        _provisional = null;
        _currentUtterance = -1;
        ScrollToEnd();
    }

    private CaptionLine AddLine(int id)
    {
        var line = new CaptionLine { UtteranceId = id };
        Lines.Add(line);
        Trim();
        return line;
    }

    private void Apply(CaptionLine line, CaptionEvent ev, bool isFinal)
    {
        line.Text = ev.Text ?? string.Empty;
        line.SpeakerId = ev.SpeakerId;
        line.SpeakerLabel = ev.Speaker;
        line.Clarity = ev.Clarity;
        line.IsSelf = ev.SpeakerId is int id && FindSpeaker(id) is { IsSelf: true };
        if (ev.StartedAt is double epoch && epoch > 0)
            line.SpokenAt = DateTimeOffset.FromUnixTimeMilliseconds((long)(epoch * 1000));
        // Assigned after Text so the inline builder overwrites the plain text, not the
        // reverse — the attached property fires on assignment, not on render.
        if (ev.Words is { Count: > 0 }) line.Words = ev.Words;
        line.IsFinal = isFinal;
    }

    private void OnDiscarded(int id)
    {
        if (_provisional is not null && _currentUtterance == id) Lines.Remove(_provisional);
        _provisional = null;
        _currentUtterance = -1;
    }

    /// <summary>Remember which line was right-clicked, so Copy knows what to copy.</summary>
    private void OnTranscriptRightTapped(object sender, RightTappedRoutedEventArgs e)
    {
        if (sender is FrameworkElement { DataContext: CaptionLine line }) _contextLine = line;
    }

    private CaptionLine? _contextLine;

    private void OnCopyLine(object sender, RoutedEventArgs e)
    {
        if (_contextLine is null) return;
        CopyToClipboard(_contextLine.ToPlainText());
    }

    private void OnCopyAll(object sender, RoutedEventArgs e) =>
        CopyToClipboard(string.Join(Environment.NewLine,
            Lines.Where(l => l.IsFinal).Select(l => l.ToPlainText())));

    private static void CopyToClipboard(string text)
    {
        if (string.IsNullOrEmpty(text)) return;
        var package = new Windows.ApplicationModel.DataTransfer.DataPackage();
        package.SetText(text);
        Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(package);
    }

    private void Trim()
    {
        while (Lines.Count > MaxLines) Lines.RemoveAt(0);
    }

    private void ScrollToEnd() =>
        CaptionScroller.ChangeView(null, CaptionScroller.ScrollableHeight, null, disableAnimation: false);

    // ---------- status ----------

    private void OnLevel(LevelEvent lv)
    {
        // Restarted even while paused: this is the proof that audio is still flowing from the
        // backend, and the recording clock leans on it to avoid counting up through a dead
        // capture thread.
        _sinceLevel.Restart();
        if (!_running) return;
        // The bar is 26px tall and fills from the bottom, so the level maps to a height
        // rather than a ProgressBar value.
        var fraction = Math.Clamp((lv.Db + 60) / 60.0, 0, 1);
        LevelFill.Height = fraction * 26.0;
    }

    private void OnStatus(StatusEvent st)
    {
        if (st.Running is bool running) SetRunning(running);
        _backendLoading = st.State == "loading";

        if (st.State == "error")
        {
            ShowActionableError(st);
            SetStatus(st.Code == "mic_denied" ? "Microphone blocked" : "Error");
            return;
        }

        // Only a successful capture clears the banner. The backend pauses itself after a
        // microphone failure and immediately reports "stopped", so clearing on every status
        // would erase the explanation milliseconds after showing it — leaving a message that
        // reads as if the user had stopped capture themselves.
        // "listening" or "stopped" both mean the engine finished loading and the pipeline is
        // up; "loading" does not. This is the only trustworthy ready signal — the socket opens
        // long before the model is usable.
        if (st.State is "listening" or "stopped")
        {
            _engineReadyThisSession = true;
            CompleteSwitchIfPending();
        }

        if (st.State == "listening")
        {
            _micProblem = false;
            // A sticky notice explains something the user did that didn't take effect, so it
            // outlives the recovery it describes — otherwise the model silently snaps back
            // with no explanation at all.
            if (!_infoSticky) MicInfoBar.IsOpen = false;
            ShowReadyState();
            ShowElapsed();
            return;
        }

        if (_micProblem && st.State == "stopped")
        {
            // Keep the real reason on screen rather than the generic paused text.
            return;
        }

        // A spinner beside "Stopped" reads as "still working on it". Once the backend reports
        // it is simply paused, the engine is up and the wait is over.
        if (st.State == "stopped") ShowIdleState();

        SetStatus(st.State switch
        {
            "loading" => $"Loading {st.Model}…",
            "stopped" => "Stopped · microphone released",
            _ => st.State,
        });
    }

    /// <summary>
    /// Hand the status line to the elapsed-time counter. The device name moves to the picker;
    /// a running clock is the more useful thing to show, because it is the one piece of state
    /// that says "the microphone is open right now".
    /// </summary>
    private void ShowElapsed()
    {
        // Repeated "listening" reports during one run must not restart the count.
        if (!_captureClock.IsRunning) _captureClock.Restart();
        // This one restarts unconditionally: a "listening" report is itself proof of life, and
        // after a reconnect the stall timer would otherwise still be carrying the whole outage
        // and cry "No audio" a second later. The one indicator that must not raise a false
        // alarm is the one claiming captions have stopped.
        _sinceLevel.Restart();
        if (!_elapsedTimer.IsEnabled) _elapsedTimer.Start();

        // No AutomationProperties.Name here: a TextBlock's UIA name defaults to its text, and
        // naming it would read out the label instead of the value — and instead of whatever
        // failure message replaces it later.
        _audioStalled = false;
        ToolTipService.SetToolTip(StatusText, RecordingTimeHint);

        StatusText.Text = FormatElapsed(_captureClock.Elapsed);
    }

    private const string RecordingTimeHint = "How long the microphone has been open";
    private const string NoAudioHint =
        "The microphone is open but no sound is reaching it. Try another input device.";

    /// <summary>Whether the status line is currently reporting a stall rather than a count.</summary>
    private bool _audioStalled;

    /// <summary>
    /// A counter that keeps climbing is read as "everything is fine", so it must not keep
    /// climbing when audio has stopped arriving. The capture thread can die while the process
    /// lives; silently animating through that is the failure this app can least afford.
    /// </summary>
    private void TickElapsed()
    {
        if (!_captureClock.IsRunning) return;

        var stalled = _sinceLevel.Elapsed > AudioStallAfter;
        if (stalled != _audioStalled)
        {
            // Only on the transition. Reassigning a tooltip under a resting pointer is what
            // stopped the per-word tooltip opening at all, and this one is the recovery hint
            // for the worst failure this app has.
            _audioStalled = stalled;
            ToolTipService.SetToolTip(StatusText, stalled ? NoAudioHint : RecordingTimeHint);
        }

        StatusText.Text = stalled ? "No audio" : FormatElapsed(_captureClock.Elapsed);
    }

    /// <summary>
    /// Any other status message owns the line, so the counter has to yield — otherwise the
    /// next tick would paint over an error the user needs to read.
    ///
    /// This stops the ticking but deliberately does NOT reset the count. A dropped socket takes
    /// this path while the microphone stays open, and restarting from zero would misreport how
    /// long the room has been recorded. The count is reset only where capture actually stops.
    /// </summary>
    private void SetStatus(string text)
    {
        _elapsedTimer.Stop();
        ToolTipService.SetToolTip(StatusText, null);
        StatusText.Text = text;
    }

    /// <summary>Capture really stopped: the next run starts from zero.</summary>
    private void ResetCaptureClock()
    {
        _elapsedTimer.Stop();
        _captureClock.Reset();
        _sinceLevel.Reset();
    }

    private static string FormatElapsed(TimeSpan elapsed) =>
        elapsed.TotalHours >= 1
            ? $"{(int)elapsed.TotalHours}:{elapsed.Minutes:00}:{elapsed.Seconds:00}"
            : $"{elapsed.Minutes}:{elapsed.Seconds:00}";

    /// <summary>
    /// Explain a failure the user can actually act on. A hard-of-hearing user staring at a
    /// blank transcript needs "microphone access is off" and a way to fix it, not a PortAudio
    /// error dump.
    /// </summary>
    private void ShowActionableError(StatusEvent st)
    {
        _micProblem = st.Code is "mic_denied" or "mic_unavailable";

        switch (st.Code)
        {
            case "mic_denied":
                _micCanPrompt = false;
                MicInfoBar.Severity = InfoBarSeverity.Warning;
                MicInfoBar.Title = "Microphone access is off";
                MicInfoBar.Message =
                    "Windows is blocking microphone access for Sunno, so nothing can " +
                    "be transcribed. Turn it on under Privacy & security › Microphone.";
                MicActionLink.Content = "Open Settings";
                MicActionLink.Visibility = Visibility.Visible;
                break;

            case "mic_unavailable":
                MicInfoBar.Severity = InfoBarSeverity.Warning;
                MicInfoBar.Title = "Microphone unavailable";
                MicInfoBar.Message =
                    (st.Message ?? "The microphone could not be opened.") +
                    " Try choosing a different microphone below.";
                MicActionLink.Visibility = Visibility.Collapsed;
                break;

            default:
                MicInfoBar.Severity = InfoBarSeverity.Error;
                MicInfoBar.Title = "Something went wrong";
                MicInfoBar.Message = st.Message ?? "Unknown error.";
                MicActionLink.Visibility = Visibility.Collapsed;
                break;
        }
        MicInfoBar.IsOpen = true;
    }

    /// <summary>
    /// The backend died. Say so plainly and point at the log — a user who is relying on this to
    /// follow a conversation must never be left watching a spinner that will never resolve.
    /// </summary>
    private void OnBackendCrashed(string message)
    {
        App.Trace($"backend crashed: {message.Split('\n')[0]}");
        _backendLoading = false;

        // A crash while switching means the new model never came up. Fall back to something
        // that actually loads instead of leaving the app dead — and never persist the choice
        // that broke it, or every future launch would reload it and crash again.
        if (_switchingTo is { } failed)
        {
            // Distinguishes a crash while the target was still downloading from one while it
            // was loading. Only the latter is evidence the target itself is bad.
            var wasLoading = _awaitingSwitchReconnect;

            _switchingTo = null;
            _awaitingSwitchReconnect = false;
            foreach (var m in Models) { m.IsBusy = false; m.Refresh(); }
            SelectModelRow(_lastGoodModel);

            var fallback = _recoveringModel ? null : PickFallbackModel(failed);
            if (fallback is not null)
            {
                _recoveringModel = true;
                if (wasLoading) RaiseFallbackNotice(failed, fallback);
                else RaiseDownloadInterruptedNotice(failed, fallback);
                _ = SwitchModelAsync(fallback);
                return;
            }
        }
        else if (!_recoveringModel && !_engineReadyThisSession && Models.Count > 0)
        {
            // Crashing before the engine has *ever* reported ready this session means the
            // stored choice itself doesn't load — the state an interrupted or unverified
            // switch can leave behind. Guarded on that rather than on the model id: a crash
            // after hours of working transcription is a runtime fault, and silently demoting
            // the user's deliberate choice because of it would be wrong.
            var fallback = PickFallbackModel(_settings.Model);
            if (fallback is not null)
            {
                _recoveringModel = true;
                RaiseFallbackNotice(_settings.Model, fallback);
                // Not persisted here: the normal completion path records it once it loads.
                _ = SwitchModelAsync(fallback);
                return;
            }
        }

        _backendFatal = true;
        ShowFatalBackendError(message);
    }

    /// <summary>
    /// Report a backend we cannot recover from. Split out so the restart path can reuse it:
    /// a failed restart has already killed the old process, so silently returning would leave
    /// "Reconnecting…" on screen forever with nothing left to reconnect to.
    /// </summary>
    private void ShowFatalBackendError(string message)
    {
        _backendFatal = true;
        _backendLoading = false;
        SetStatus("Speech engine stopped");
        _micProblem = false;
        _micCanPrompt = false;
        _infoSticky = false;
        MicInfoBar.Severity = InfoBarSeverity.Error;
        MicInfoBar.Title = "The speech engine stopped";
        MicInfoBar.Message = $"{message}\n\nDetails were written to {BackendHost.DisplayLogPath}";
        MicActionLink.Content = "Copy details";
        MicActionLink.Visibility = Visibility.Visible;
        MicInfoBar.IsOpen = true;
        _crashDetail = $"{message}\n\nLog: {BackendHost.DisplayLogPath}";
    }

    private string? _crashDetail;

    /// <summary>Explain a demotion the user didn't ask for, and keep it on screen.</summary>
    private void RaiseFallbackNotice(string failedId, string fallbackId)
    {
        var failedName = Models.FirstOrDefault(m => m.Id == failedId)?.Name ?? failedId;
        var fallbackName = Models.FirstOrDefault(m => m.Id == fallbackId)?.Name ?? fallbackId;

        MicInfoBar.Severity = InfoBarSeverity.Warning;
        MicInfoBar.Title = $"{failedName} couldn't be loaded";
        MicInfoBar.Message = $"Using {fallbackName} instead.";
        MicActionLink.Visibility = Visibility.Collapsed;
        MicInfoBar.IsOpen = true;
        _infoSticky = true;
    }

    /// <summary>
    /// The engine died while the chosen model was still downloading, so the model itself is not
    /// implicated — saying it "couldn't be loaded" would be a guess, and a wrong one.
    /// </summary>
    private void RaiseDownloadInterruptedNotice(string targetId, string fallbackId)
    {
        var targetName = Models.FirstOrDefault(m => m.Id == targetId)?.Name ?? targetId;
        var fallbackName = Models.FirstOrDefault(m => m.Id == fallbackId)?.Name ?? fallbackId;

        MicInfoBar.Severity = InfoBarSeverity.Warning;
        MicInfoBar.Title = $"Download of {targetName} was interrupted";
        MicInfoBar.Message = $"Still using {fallbackName}. Selecting it again resumes the download.";
        MicActionLink.Visibility = Visibility.Collapsed;
        MicInfoBar.IsOpen = true;
        _infoSticky = true;
    }

    private async void OnMicAction(object sender, RoutedEventArgs e)
    {
        // The same button means different things depending on whether Windows will still
        // prompt: asking again is useless once the answer has been recorded.
        if (_crashDetail is not null)
        {
            var data = new Windows.ApplicationModel.DataTransfer.DataPackage();
            data.SetText(_crashDetail);
            Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(data);
            return;
        }
        if (_micCanPrompt)
        {
            // No OS round trip in either branch. A declined dialog was our own decision, not
            // the OS's, and RequestAccessAsync is a proven no-op for a full-trust packaged
            // app — it returns Allowed without prompting. Re-querying consent here also
            // reintroduces the repeat-CheckAccess call that killed the process.
            _micDeclined = false;
            ApplyMicrophoneStatus(_micStatus);
            return;
        }
        await Windows.System.Launcher.LaunchUriAsync(
            new Uri("ms-settings:privacy-microphone"));
    }

    /// <summary>
    /// Settle microphone consent.
    ///
    /// Windows will not raise its own dialog here: a runFullTrust packaged app is granted the
    /// microphone by default and RequestAccessAsync returns Allowed without prompting (verified
    /// against a live package for the never-asked, Prompt and Deny consent states). The
    /// Camera-style system prompt is an AppContainer behaviour, and full trust is non-negotiable
    /// because CUDA cannot load inside a container. So we ask once ourselves — which also lets
    /// us say the thing that actually reassures people, that audio never leaves the machine.
    /// </summary>
    private async Task EnsureMicrophoneAccessAsync()
    {
        // Deliberately reuses the status read during construction rather than calling
        // CheckAccess again. A second call from inside the Loaded handler reproducibly took
        // the process down with a stowed exception in Microsoft.UI.Xaml, and re-reading buys
        // nothing: consent cannot change in the ~100 ms between the two, and AccessChanged
        // covers any change afterwards.
        var status = _micStatus;
        App.Trace($"mic: reusing status={status}");

        if (!_settings.MicrophoneAsked && status is null or AppCapabilityAccessStatus.Allowed)
            await AskForMicrophoneAsync();

        ApplyMicrophoneStatus(status);

        // Notice a later grant from Settings, so the fallback isn't a dead end that needs a relaunch.
        MicrophoneAccess.Changed += OnMicAccessChanged;
        App.Trace("mic: done");
    }

    /// <summary>Ask once, in two sentences, with the honest answer to "where does my voice go".</summary>
    private async Task AskForMicrophoneAsync()
    {
        _settings.MicrophoneAsked = true;
        _settings.Save();

        var dialog = new ContentDialog
        {
            XamlRoot = Content.XamlRoot,
            Title = "Use your microphone?",
            Content = "Sunno listens to your microphone to caption what people say. " +
                      "Audio is transcribed on this PC and never leaves it.",
            PrimaryButtonText = "Allow",
            CloseButtonText = "Not now",
            DefaultButton = ContentDialogButton.Primary,
        };

        if (await dialog.ShowAsync() == ContentDialogResult.Primary) return;

        // Declining is a real answer, so honour it rather than quietly listening anyway.
        _micDeclined = true;
        _micProblem = true;
        _micCanPrompt = true;
        MicInfoBar.Severity = InfoBarSeverity.Informational;
        MicInfoBar.Title = "Microphone is off";
        MicInfoBar.Message = "Sunno isn't listening. Turn it on whenever you're ready.";
        MicActionLink.Content = "Turn on";
        MicActionLink.Visibility = Visibility.Visible;
        MicInfoBar.IsOpen = true;
    }

    private bool _micDeclined;

    private void OnMicAccessChanged() =>
        _ui.TryEnqueue(() =>
        {
            // The event itself is the signal that access changed; re-reading is what the
            // event exists to make safe, and by here we are off the Loaded path.
            _micStatus = MicrophoneAccess.Check();
            ApplyMicrophoneStatus(_micStatus);
        });

    /// <summary>
    /// Render a consent status. Each denial has a different remedy, so they can't share one
    /// message: pointing at the per-app toggle is actively misleading when the device-wide one
    /// is off, because the per-app control isn't even shown in that state.
    /// </summary>
    private void ApplyMicrophoneStatus(AppCapabilityAccessStatus? status)
    {
        App.Trace($"ApplyMicrophoneStatus({status}) declined={_micDeclined}");
        // Null means no package identity (an unpackaged dev build). There is no consent state
        // to honour, so let the backend's runtime error be the only signal.
        if (status is null or AppCapabilityAccessStatus.Allowed)
        {
            // An explicit "not now" outranks the OS default, which for a full-trust app is
            // always Allowed and would otherwise silently overturn the user's own answer.
            if (_micDeclined) return;

            _micProblem = false;
            _micCanPrompt = false;
            MicInfoBar.IsOpen = false;
            _micGranted = true;
            TryStartCapture();
            return;
        }

        _micProblem = true;
        MicInfoBar.Severity = InfoBarSeverity.Warning;
        MicActionLink.Visibility = Visibility.Visible;

        switch (status)
        {
            case AppCapabilityAccessStatus.UserPromptRequired:
                // RequestAccessAsync is documented never to return this, so arriving here means
                // the prompt could not be shown. Offer it as an explicit action rather than
                // pretending the user refused.
                _micCanPrompt = true;
                MicInfoBar.Title = "Allow microphone access";
                MicInfoBar.Message =
                    "Sunno needs your microphone to transcribe what people say. " +
                    "Audio is processed on this PC and never leaves it.";
                break;

            case AppCapabilityAccessStatus.DeniedByUser:
                _micCanPrompt = false;
                MicInfoBar.Title = "Microphone access is off";
                MicInfoBar.Message =
                    "Microphone access for Sunno is turned off, so nothing can be " +
                    "transcribed. Turn it back on under Privacy & security › Microphone.";
                break;

            case AppCapabilityAccessStatus.DeniedBySystem:
                _micCanPrompt = false;
                MicInfoBar.Title = "Microphone is off for this device";
                MicInfoBar.Message =
                    "Microphone access is turned off for the whole device, or for all desktop " +
                    "apps, so no app can transcribe. Turn it on under Privacy & security › " +
                    "Microphone.";
                break;

            default:
                // NotDeclaredByApp: a packaging defect, not something the user can fix.
                _micCanPrompt = false;
                MicInfoBar.Severity = InfoBarSeverity.Error;
                MicInfoBar.Title = "Microphone capability missing";
                MicInfoBar.Message =
                    "This build didn't declare the microphone capability, so Windows won't " +
                    "grant access. Reinstalling from a complete package should fix it.";
                MicActionLink.Visibility = Visibility.Collapsed;
                break;
        }

        MicActionLink.Content = _micCanPrompt ? "Allow" : "Open Settings";
        MicInfoBar.IsOpen = true;
    }

    /// <summary>
    /// Begin capture once consent and the WebSocket are both ready.
    ///
    /// Either can finish first — granting takes a couple of seconds while the model takes ~33 s
    /// — and <see cref="CaptionClient"/> drops sends on a closed socket, so both paths call
    /// this and whichever completes last actually starts capture.
    /// </summary>
    private void TryStartCapture()
    {
        if (!_micGranted || _micDeclined || !_startedPaused || _captureRequested || !_connected) return;
        _captureRequested = true;
        _ = _client.StartCaptureAsync();
    }

    /// <summary>Populate the picker lazily — the catalogue costs a Hub round trip.</summary>
    /// <summary>
    /// Commit a pending switch, once the engine has actually reported itself ready.
    ///
    /// Not driven off the socket: the backend serves WebSocket clients while the model is still
    /// loading, so connecting proves nothing about whether the chosen model works.
    /// </summary>
    private void CompleteSwitchIfPending()
    {
        if (!_awaitingSwitchReconnect || _switchingTo is not { } finished) return;

        // Captured before reset: a recovery's own completion must not dismiss the notice that
        // explains the recovery, or the demotion becomes silent again.
        var wasRecovery = _recoveringModel;

        _awaitingSwitchReconnect = false;
        _switchingTo = null;
        _recoveringModel = false;

        // The engine loaded it, so this is now the choice worth remembering.
        _lastGoodModel = finished;
        _settings.Model = finished;
        _settings.Save();

        var row = Models.FirstOrDefault(m => m.Id == finished);
        if (row is not null)
        {
            row.Available = true;
            row.Refresh();
        }
        SelectModelRow(finished);
        foreach (var other in Models.Where(m => m.Id != finished)) other.Refresh();

        // A switch the user asked for supersedes whatever the last notice explained; a fallback
        // does not, because the notice is about the fallback itself.
        if (_infoSticky && !wasRecovery)
        {
            _infoSticky = false;
            MicInfoBar.IsOpen = false;
        }
    }

    /// <summary>
    /// Choose something to fall back to that is actually on disk. Falling back to a model that
    /// still needs a multi-gigabyte download would trade a crash for a download prompt, and
    /// persisting it would make every future launch open that prompt instead of captioning.
    /// Null means nothing can be chosen honestly.
    /// </summary>
    private string? PickFallbackModel(string failed)
    {
        if (_lastGoodModel.Length > 0 && _lastGoodModel != failed)
        {
            var known = Models.FirstOrDefault(m => m.Id == _lastGoodModel);
            // No catalogue yet means we crashed before it arrived; trust the last good model,
            // which by definition loaded at some point.
            if (known is null || known.Available) return _lastGoodModel;
        }

        return Models.FirstOrDefault(m => m.Available && m.Id != failed)?.Id;
    }

    private void OnMicInfoClosed(InfoBar sender, object args) => _infoSticky = false;

    private bool _modelSectionOpen;

    /// <summary>
    /// Open or close the model list with a single eased height animation.
    ///
    /// Layout properties need <c>EnableDependentAnimation</c> — without it the animation is
    /// silently dropped and the panel jumps, which is the exact jarring motion this replaced.
    /// The list is measured while collapsed because the ScrollViewer gives its child unbounded
    /// height, so the target is known before the animation starts.
    /// </summary>
    private void OnToggleModelSection(object sender, RoutedEventArgs e)
    {
        _modelSectionOpen = !_modelSectionOpen;

        var target = 0.0;
        if (_modelSectionOpen)
        {
            ModelOptions.Measure(new Windows.Foundation.Size(
                ModelPanel.ActualWidth > 0 ? ModelPanel.ActualWidth : 240, double.PositiveInfinity));
            target = ModelOptions.DesiredSize.Height;
        }

        AnimateModelPanel(target);
        AnimateChevron(_modelSectionOpen ? 180 : 0);
    }

    private void AnimateModelPanel(double toHeight)
    {
        var animation = new DoubleAnimation
        {
            To = toHeight,
            Duration = new Duration(TimeSpan.FromMilliseconds(220)),
            EnableDependentAnimation = true,
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut },
        };
        Storyboard.SetTarget(animation, ModelPanel);
        Storyboard.SetTargetProperty(animation, "Height");

        var story = new Storyboard();
        story.Children.Add(animation);
        story.Begin();
    }

    private void AnimateChevron(double angle)
    {
        var animation = new DoubleAnimation
        {
            To = angle,
            Duration = new Duration(TimeSpan.FromMilliseconds(220)),
            EnableDependentAnimation = true,
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut },
        };
        Storyboard.SetTarget(animation, ModelChevronRotate);
        Storyboard.SetTargetProperty(animation, "Angle");

        var story = new Storyboard();
        story.Children.Add(animation);
        story.Begin();
    }

    /// <summary>
    /// Keep an open panel sized to its content, so a row changing height mid-download doesn't
    /// leave the list clipped or floating above empty space.
    /// </summary>
    private void ResizeModelPanelIfOpen()
    {
        if (!_modelSectionOpen) return;
        ModelOptions.Measure(new Windows.Foundation.Size(
            ModelPanel.ActualWidth > 0 ? ModelPanel.ActualWidth : 240, double.PositiveInfinity));
        ModelPanel.Height = ModelOptions.DesiredSize.Height;
    }

    /// <summary>Keep the collapsed header's summary in step with what's actually loaded.</summary>
    private void UpdateHeaderModelName()
    {
        var active = Models.FirstOrDefault(m => m.IsSelected);
        HeaderModelName.Text = active?.Name ?? string.Empty;
        // The header trims too, and it's the only thing visible while collapsed.
        ToolTipService.SetToolTip(HeaderModelName, active?.Tooltip);
    }

    private void OnModelCatalog(string current, IReadOnlyList<ModelOption> options)
    {
        _suppressModelEvent = true;
        try
        {
            Models.Clear();
            foreach (var o in options)
            {
                var row = new ModelRow
                {
                    Id = o.Id,
                    Name = o.Name,
                    Detail = o.Detail,
                    ApproxMb = o.ApproxMb,
                    Available = o.Available,
                    LagText = o.LagText,
                    Responsive = o.Responsive,
                    IsSelected = o.Id == current,
                    InUse = o.Id == current,
                };
                row.Refresh();
                Models.Add(row);
            }
        }
        finally
        {
            _suppressModelEvent = false;
        }
        UpdateHeaderModelName();
        // The catalogue just arrived or changed, so an open panel's content height moved.
        ResizeModelPanelIfOpen();
    }

    /// <summary>
    /// Switch models. Downloading first when needed, then reloading the engine.
    ///
    /// The engine is built once at backend startup, so this restarts the child process rather
    /// than swapping in place. The transcript is UI-side state and speaker profiles are
    /// persisted server-side, so the only real cost is the reload itself.
    /// </summary>
    private async void OnModelChecked(object sender, RoutedEventArgs e)
    {
        if (_suppressModelEvent) return;
        if (sender is not RadioButton { Tag: string id }) return;

        var row = Models.FirstOrDefault(m => m.Id == id);
        if (row is null || id == _settings.Model || _switchingTo is not null) return;

        // Claimed up front so a second click is ignored while the download runs, which can
        // take minutes; SwitchModelAsync re-asserts it for the paths that don't come via here.
        _switchingTo = id;

        if (!row.Available)
        {
            // Downloading runs on the live backend, which already reports byte progress; the
            // engine only reloads once the bytes are on disk.
            row.ShowProgress(0);
            await _client.DownloadModelAsync(id);
            return;   // OnDownloadComplete resumes the switch
        }

        await SwitchModelAsync(id);
    }

    private async Task SwitchModelAsync(string id)
    {
        var row = Models.FirstOrDefault(m => m.Id == id);
        row?.ShowLoading();

        // Set here rather than at every call site, so the recovery paths can't forget it and
        // leave OnConnection unable to recognise the switch completing.
        _switchingTo = id;

        ShowLoadingState($"Loading {row?.Name ?? id}");

        // Reset the capture handshake for the new process, and keep honouring a declined
        // microphone — a model switch must not become a back door into opening it.
        _captureRequested = false;
        _connected = false;
        _engineReadyThisSession = false;
        _startedPaused = !_micGranted || _micDeclined;
        _awaitingSwitchReconnect = true;
        // The backend process is about to be replaced, so capture genuinely restarts.
        ResetCaptureClock();

        // Deliberately NOT persisted yet. A model that downloads but fails to load would
        // otherwise become the choice reloaded on every future launch, turning one bad switch
        // into a crash loop with no way out from inside the app.
        var error = _backend.Restart(
            device: _settings.DeviceIndex?.ToString(),
            model: id,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused,
            loopbackDevice: _settings.LoopbackDeviceIndex);

        if (!string.IsNullOrEmpty(error))
        {
            _switchingTo = null;
            _awaitingSwitchReconnect = false;
            _recoveringModel = false;
            if (row is not null) { row.IsBusy = false; row.Refresh(); }
            SelectModelRow(_lastGoodModel);
            // Restart already killed the old backend before failing to start the new one, so
            // nothing is running. Reporting it as fatal beats an endless "Reconnecting…" to a
            // process that no longer exists.
            ShowFatalBackendError(error);
            return;
        }

        // The socket drops with the old process; the client's own retry loop reconnects.
        await Task.CompletedTask;
    }

    /// <summary>
    /// The empty state has two jobs: "nothing said yet" and "not ready yet". Only the second
    /// deserves a spinner, and conflating them promises captions the engine cannot yet produce.
    /// </summary>
    private void ShowLoadingState(string title)
    {
        LoadingRing.IsActive = true;
        LoadingRing.Visibility = Visibility.Visible;
        EmptyGlyph.Visibility = Visibility.Collapsed;
        EmptyTitle.Text = title;
        EmptyDetail.Text = "This takes about half a minute.";
        EmptyState.Visibility = Lines.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    private void ShowReadyState()
    {
        LoadingRing.IsActive = false;
        LoadingRing.Visibility = Visibility.Collapsed;
        EmptyGlyph.Visibility = Visibility.Visible;
        EmptyTitle.Text = "Listening for speech";
        EmptyDetail.Text = "Captions appear here as people talk.";
    }

    /// <summary>Engine is up, but capture is paused — by the user, or awaiting consent.</summary>
    private void ShowIdleState()
    {
        LoadingRing.IsActive = false;
        LoadingRing.Visibility = Visibility.Collapsed;
        EmptyGlyph.Visibility = Visibility.Visible;
        EmptyTitle.Text = "Not listening";
        EmptyDetail.Text = "Press the microphone button to start.";
    }

    private void OnConnection(bool connected)
    {
        _connected = connected;
        if (connected)
        {
            _backendLoading = false;
            // Consent may have been granted while the model was still loading; sends are
            // dropped on a closed socket, so this is the other half of that handshake.
            TryStartCapture();

            // Deliberately NOT where a switch is completed. The backend accepts WebSocket
            // connections before it loads the engine, so "connected" arrives roughly half a
            // minute before "usable" — committing here would persist a model that has not
            // actually loaded yet. Completion is driven by the first ready status instead.
            if (Models.Count == 0)
            {
                // The picker is always visible now, so it needs its contents up front.
                _ = _client.RequestModelsAsync();
            }
            return;
        }
        // A dead backend also looks "disconnected", and its reconnect attempts would otherwise
        // paint over the real explanation with a reassuring one.
        if (_backendFatal) return;
        // On a cold start the socket isn't up yet because the model is still loading.
        // "Starting…" is more truthful than "Reconnecting…" for a first run.
        SetStatus(_backendLoading ? "Starting the speech engine…" : "Reconnecting…");
    }

    private void OnRoster(IReadOnlyList<SpeakerInfo> speakers)
    {
        Speakers.Clear();
        foreach (var s in speakers)
        {
            Speakers.Add(new SpeakerRow
            {
                Id = s.Id,
                Label = s.IsSelf ? $"{s.Label} (You)" : s.Label,
                IsSelf = s.IsSelf,
                Named = s.Named,
            });
        }

        NoSpeakers.Visibility = Speakers.Count == 0 ? Visibility.Visible : Visibility.Collapsed;

        // Re-label existing lines so history stays consistent after a rename or merge.
        foreach (var line in Lines)
        {
            if (line.SpeakerId is not int id) continue;
            var info = speakers.FirstOrDefault(s => s.Id == id);
            if (info is null) continue;
            line.SpeakerLabel = info.Label;
            line.IsSelf = info.IsSelf;
        }
    }

    private SpeakerRow? FindSpeaker(int id) => Speakers.FirstOrDefault(s => s.Id == id);

    private void SetRunning(bool running)
    {
        _running = running;
        ToggleGlyph.Glyph = running ? "\uE71A" : "\uE720";   // stop square / microphone
        ToggleButton.SetValue(AutomationProperties.NameProperty,
            running ? "Stop transcribing and release the microphone" : "Start transcribing");
        ToolTipService.SetToolTip(ToggleButton,
            running ? "Stop transcribing (Space)" : "Start transcribing (Space)");

        if (!running)
        {
            // Capture really stopped; the status message that follows owns the line, and the
            // next run starts from zero.
            ResetCaptureClock();
            LevelFill.Height = 0;
            if (_provisional is not null) Lines.Remove(_provisional);
            _provisional = null;
            _currentUtterance = -1;
        }
    }

    // ---------- first-run setup ----------

    private void OnModelRequired(IReadOnlyList<ModelOption> options)
    {
        ModelList.Items.Clear();
        foreach (var o in options)
            ModelList.Items.Add(new ModelChoice(o.Id, o.Name, o.Detail, o.ApproxMb, o.Available,
                                                o.LagText, o.Responsive));

        // Preselect the model this hardware can actually keep up with, preferring one
        // already on disk. Picking purely by "already downloaded" would start a CPU-only
        // machine on whatever happened to be cached, which may be the slowest option.
        var preferred = ModelList.Items
            .OfType<ModelChoice>()
            .Where(m => m.Responsive)
            .OrderByDescending(m => m.Available)
            .FirstOrDefault()
            ?? ModelList.Items.OfType<ModelChoice>().FirstOrDefault(m => m.Available);
        ModelList.SelectedItem = preferred ?? ModelList.Items.FirstOrDefault();

        SetupError.IsOpen = false;
        DownloadPanel.Visibility = Visibility.Collapsed;
        DownloadButton.IsEnabled = true;
        SetupOverlay.Visibility = Visibility.Visible;
        SetStatus("Setup required");
    }

    private async void OnDownloadModel(object sender, RoutedEventArgs e)
    {
        if (ModelList.SelectedItem is not ModelChoice choice) return;

        SetupError.IsOpen = false;
        DownloadButton.IsEnabled = false;
        ModelList.IsEnabled = false;
        DownloadPanel.Visibility = Visibility.Visible;
        DownloadBar.IsIndeterminate = true;
        DownloadText.Text = choice.Available
            ? "Preparing…"
            : $"Starting download of {choice.SizeLabel}…";

        // Remember the choice now, so a restart mid-download resumes with the same model
        // rather than asking again.
        _settings.Model = choice.Id;
        _settings.Save();

        await _client.DownloadModelAsync(choice.Id);
    }

    private void OnDownloadProgress(DownloadProgressEvent p)
    {
        if (_switchingTo is not null)
        {
            var row = Models.FirstOrDefault(m => m.Id == p.Model);
            row?.ShowProgress(p.Percent);
            return;
        }

        DownloadBar.IsIndeterminate = false;
        DownloadBar.Value = Math.Clamp(p.Percent, 0, 100);
        DownloadText.Text =
            $"{p.Downloaded / 1048576.0:0} MB of {p.Total / 1048576.0:0} MB · {p.Percent:0}%";
    }

    private void OnDownloadComplete()
    {
        if (_switchingTo is { } pending)
        {
            var row = Models.FirstOrDefault(m => m.Id == pending);
            if (row is not null) row.Available = true;
            _ = SwitchModelAsync(pending);
            return;
        }

        DownloadBar.Value = 100;
        DownloadText.Text = "Done. Loading the speech engine…";
        SetupOverlay.Visibility = Visibility.Collapsed;
        ModelList.IsEnabled = true;
    }

    private void OnDownloadFailed(string message)
    {
        if (_switchingTo is { } pending)
        {
            // Abandon the switch and put the radio back where it was, so the list keeps telling
            // the truth about which model is actually loaded.
            var row = Models.FirstOrDefault(m => m.Id == pending);
            if (row is not null) { row.IsBusy = false; row.Refresh(); }
            _switchingTo = null;
            SelectModelRow(_settings.Model);

            MicInfoBar.Severity = InfoBarSeverity.Warning;
            MicInfoBar.Title = "Couldn't download that model";
            MicInfoBar.Message = message;
            MicActionLink.Visibility = Visibility.Collapsed;
            MicInfoBar.IsOpen = true;
            return;
        }

        SetupError.Title = "Download failed";
        SetupError.Message = message;
        SetupError.IsOpen = true;
        DownloadPanel.Visibility = Visibility.Collapsed;
        DownloadButton.IsEnabled = true;
        ModelList.IsEnabled = true;
    }

    /// <summary>Move the radio selection without re-entering the Checked handler.</summary>
    private void SelectModelRow(string id)
    {
        _suppressModelEvent = true;
        try
        {
            foreach (var m in Models)
            {
                m.SetSelected(m.Id == id);
                m.InUse = m.Id == id;
            }
        }
        finally
        {
            _suppressModelEvent = false;
        }
        UpdateHeaderModelName();
    }

    // ---------- devices ----------

    private async Task LoadDevicesAsync()
    {
        App.Trace("LoadDevicesAsync start");
        // The backend needs a moment to bind its HTTP port on a cold start.
        for (var attempt = 0; attempt < 40; attempt++)
        {
            List<AudioDevice>? devices = null;
            try
            {
                var json = await _http.GetStringAsync("http://127.0.0.1:8765/devices.json");
                devices = ParseDevices(json);
            }
            catch
            {
                await Task.Delay(500);
                continue;
            }

            if (devices is { Count: > 0 }) _ui.TryEnqueue(() => PopulateDevices(devices));
            return;
        }
    }

    /// <summary>
    /// Fully materialises the JSON before returning.
    ///
    /// JsonElement is a view over its JsonDocument's pooled buffer, so handing elements to a
    /// callback that runs later — after the `using` disposes the document — throws on the UI
    /// thread and takes the app down. Copy to plain records here instead.
    /// </summary>
    private static List<AudioDevice> ParseDevices(string json)
    {
        var result = new List<AudioDevice>();
        using var doc = JsonDocument.Parse(json);
        if (!doc.RootElement.TryGetProperty("devices", out var arr)) return result;

        foreach (var d in arr.EnumerateArray())
        {
            var index = d.TryGetProperty("index", out var i) ? i.GetInt32() : -1;
            var name = d.TryGetProperty("name", out var n) ? n.GetString()?.Trim() : null;
            var api = d.TryGetProperty("hostapi", out var h) ? h.GetString() : null;
            var loopback = d.TryGetProperty("loopback", out var l) && l.ValueKind == JsonValueKind.True;
            if (index >= 0 && !string.IsNullOrEmpty(name))
                result.Add(new AudioDevice(index, name!, api ?? string.Empty, loopback));
        }
        return result;
    }

    private void PopulateDevices(List<AudioDevice> devices)
    {
        App.Trace($"PopulateDevices: {devices.Count}");
        _suppressDeviceEvent = true;
        try
        {
            DevicePicker.Items.Clear();

            // The same physical device is exposed once per host API (WASAPI, MME, DirectSound,
            // WDM-KS), and each API mangles the name differently — MME truncates at 31
            // characters, so one microphone arrives as "Microphone (Umik-1  Gain: 18dB",
            // "…18dB  )" and "…18dB)". Comparing letters and digits only, and treating a
            // truncated name as the same device, collapses them. The server sorts WASAPI first,
            // so the entry kept is the modern endpoint with the full name.
            var mics = new List<DeviceEntry>();
            var speakers = new List<DeviceEntry>();

            foreach (var d in devices)
            {
                var label = CleanDeviceName(d.Name);
                var key = DeviceKey(label);
                if (key.Length == 0) continue;
                if (IsDefaultAlias(label)) continue;

                var group = d.Loopback ? speakers : mics;
                var match = group.FirstOrDefault(e => IsSameDevice(e, key, d.Name.Length));
                if (match is not null)
                {
                    // Remember the index anyway: the saved device may be one of the duplicates
                    // that lost, and the picker still has to show it as selected.
                    match.Aliases.Add(d.Index);
                    continue;
                }

                group.Add(new DeviceEntry(d with { Name = label }, key, d.Name.Length));
            }

            AddDeviceGroup("Input Device - Microphone", mics);
            AddDeviceGroup("Input Device - System Audio", speakers);
            SelectActiveDevice();
        }
        finally
        {
            _suppressDeviceEvent = false;
        }
    }

    /// <summary>One kept device plus the indices of the duplicates it absorbed.</summary>
    private sealed class DeviceEntry(AudioDevice device, string key, int rawNameLength)
    {
        public AudioDevice Device { get; } = device;
        public string Key { get; } = key;
        public int RawNameLength { get; } = rawNameLength;
        public List<int> Aliases { get; } = [device.Index];
    }

    /// <summary>
    /// Shows which device is actually being captured. The status line now counts recording
    /// time instead of naming the device, so without this nothing on screen would say what the
    /// app is listening to.
    /// </summary>
    private void SelectActiveDevice()
    {
        var wanted = _settings.LoopbackDeviceIndex ?? _settings.DeviceIndex;
        var loopback = _settings.LoopbackDeviceIndex is not null;
        if (wanted is null) return;

        foreach (var item in DevicePicker.Items.OfType<ComboBoxItem>())
        {
            if (item.Tag is not DeviceEntry entry) continue;
            if (entry.Device.Loopback != loopback) continue;
            if (!entry.Aliases.Contains(wanted.Value)) continue;
            DevicePicker.SelectedItem = item;
            // The closed picker truncates; the full name is only otherwise visible with the
            // list open, and the status line no longer carries it.
            ToolTipService.SetToolTip(DevicePicker, entry.Device.Loopback
                ? $"Captioning system audio from {entry.Device.Name}"
                : $"Captioning the microphone {entry.Device.Name}");
            return;
        }
    }

    private void AddDeviceGroup(string header, List<DeviceEntry> group)
    {
        if (group.Count == 0) return;

        // A disabled item is the only way to get a non-selectable header into a ComboBox;
        // it is skipped by keyboard navigation as well as by pointer.
        DevicePicker.Items.Add(new ComboBoxItem
        {
            Content = header,
            IsEnabled = false,
            FontSize = 11,
            FontWeight = Microsoft.UI.Text.FontWeights.SemiBold,
        });

        foreach (var entry in group)
        {
            var d = entry.Device;
            var item = new ComboBoxItem { Content = d.Name, Tag = entry };
            DevicePicker.Items.Add(item);
            ToolTipService.SetToolTip(item, d.Loopback
                ? $"Caption whatever is played through {d.Name} — calls, video, music"
                : $"{d.Name} — {d.HostApi}");
        }
    }

    /// <summary>Letters and digits only, lower-cased — immune to spacing and stray brackets.</summary>
    private static string DeviceKey(string name)
    {
        var sb = new System.Text.StringBuilder(name.Length);
        foreach (var c in name)
            if (char.IsLetterOrDigit(c)) sb.Append(char.ToLowerInvariant(c));
        return sb.ToString();
    }

    /// <summary>
    /// Length MME truncates device names to — MAXPNAMELEN is 32 including the terminator.
    /// </summary>
    private const int MmeNameLimit = 31;

    /// <summary>
    /// Whether a candidate is the device an existing entry already represents. Equal keys are
    /// the same device. A prefix match only counts when the shorter name sits exactly on MME's
    /// truncation boundary, which is the only reason a name would be cut short: a name that is
    /// merely long was not truncated, so "USB Audio Device" and "USB Audio Device Pro" stay
    /// separate rather than being silently merged into one.
    /// </summary>
    private static bool IsSameDevice(DeviceEntry kept, string key, int rawNameLength)
    {
        if (kept.Key == key) return true;

        if (key.Length < kept.Key.Length)
            return rawNameLength == MmeNameLimit && kept.Key.StartsWith(key, StringComparison.Ordinal);

        return kept.RawNameLength == MmeNameLimit && key.StartsWith(kept.Key, StringComparison.Ordinal);
    }

    /// <summary>
    /// Windows sometimes hands back an unresolved resource string for Bluetooth endpoints,
    /// e.g. "Headset (@System32\drivers\bthhfenum.sys,#2;%1 Hands-Free%0 ;(R-Phonak hearing
    /// aid))". The useful parts are the prefix and the innermost parenthesised device name.
    /// Runs of whitespace are collapsed too, because some drivers pad their names — the UMIK-1
    /// arrives as "Microphone (Umik-1  Gain: 18dB  )".
    /// </summary>
    private static string CleanDeviceName(string name)
    {
        var trimmed = Collapse(name);
        if (!trimmed.Contains('@')) return trimmed;

        var open = trimmed.LastIndexOf('(');
        var close = trimmed.IndexOf(')', open + 1);
        if (open < 0 || close <= open + 1) return trimmed;

        var inner = trimmed[(open + 1)..close].Trim();
        if (inner.Length == 0) return trimmed;

        var firstOpen = trimmed.IndexOf('(');
        var prefix = firstOpen > 0 ? trimmed[..firstOpen].Trim() : string.Empty;
        return prefix.Length > 0 ? $"{prefix} ({inner})" : inner;
    }

    /// <summary>Squeezes whitespace runs — including around brackets — down to single spaces.</summary>
    private static string Collapse(string name)
    {
        var sb = new System.Text.StringBuilder(name.Length);
        var space = false;
        foreach (var c in name)
        {
            if (char.IsWhiteSpace(c)) { space = true; continue; }
            if (space && sb.Length > 0 && c != ')') sb.Append(' ');
            space = false;
            sb.Append(c);
        }
        return sb.ToString();
    }

    /// <summary>
    /// MME and DirectSound each publish an alias for "whatever the system default is". They are
    /// not devices, their names say nothing a user would recognise, and leaving the picker on
    /// its "Default microphone" placeholder already selects the default — so they are noise in
    /// a list someone has to choose from quickly.
    /// </summary>
    private static bool IsDefaultAlias(string name) =>
        name.StartsWith("Microsoft Sound Mapper", StringComparison.OrdinalIgnoreCase) ||
        name.StartsWith("Primary Sound Capture Driver", StringComparison.OrdinalIgnoreCase) ||
        name.StartsWith("Primary Sound Driver", StringComparison.OrdinalIgnoreCase);

    private void OnDeviceChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressDeviceEvent) return;
        if (DevicePicker.SelectedItem is not ComboBoxItem { Tag: DeviceEntry entry }) return;
        var device = entry.Device;

        ToolTipService.SetToolTip(DevicePicker, device.Loopback
            ? $"Captioning system audio from {device.Name}"
            : $"Captioning the microphone {device.Name}");

        _settings.DeviceIndex = device.Loopback ? null : device.Index;
        _settings.LoopbackDeviceIndex = device.Loopback ? device.Index : null;
        _settings.Save();

        // Switching capture device means restarting the backend; the model reload is the slow
        // part, so say so rather than appear hung.
        //
        // Restart, never Dispose+Start. Dispose tears down the job object permanently and
        // latches _stopping, so a Start afterwards leaves the new capture process untied to
        // kill-on-close (it would outlive a killed UI still holding the microphone) and with
        // crash reporting silently dead for the rest of the session.
        SetStatus(device.Loopback ? "Switching to system audio…" : "Switching microphone…");
        ResetCaptureClock();

        _captureRequested = false;
        _connected = false;
        _engineReadyThisSession = false;
        _startedPaused = !_micGranted || _micDeclined;

        var error = _backend.Restart(
            device: device.Loopback ? null : device.Index.ToString(),
            model: _settings.Model,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused,
            loopbackDevice: device.Loopback ? device.Index : null);
        if (!string.IsNullOrEmpty(error)) ShowFatalBackendError(error);
    }

    // ---------- commands ----------

    /// <summary>
    /// Starting capture by hand is consent in its own right. Without this, a user who chose
    /// "Not now" and later pressed the button would be captioning while the app still believed
    /// it was declined — and the next model switch would silently stop captions again.
    /// </summary>
    private async void OnToggleCapture(object sender, RoutedEventArgs e)
    {
        if (!_running)
        {
            _micDeclined = false;
            _micGranted = true;
            if (_micProblem && MicInfoBar.Severity == InfoBarSeverity.Informational)
            {
                _micProblem = false;
                MicInfoBar.IsOpen = false;
            }
        }
        await _client.ToggleAsync();
    }

    private void OnBigger(object sender, RoutedEventArgs e) => SetFontSize(CaptionSize + 3);
    private void OnSmaller(object sender, RoutedEventArgs e) => SetFontSize(CaptionSize - 3);

    private void SetFontSize(double size)
    {
        CaptionSize = Math.Clamp(size, MinFont, MaxFont);
        // Re-materialise so the templates pick up the new size.
        var snapshot = Lines.ToList();
        Lines.Clear();
        foreach (var l in snapshot) Lines.Add(l);
        ScrollToEnd();
    }

    private void OnToggleAlwaysOnTop(object sender, RoutedEventArgs e)
    {
        if (AppWindow.Presenter is OverlappedPresenter presenter)
            presenter.IsAlwaysOnTop = AlwaysOnTopItem.IsChecked;
    }

    private void OnClear(object sender, RoutedEventArgs e)
    {
        Lines.Clear();
        _provisional = null;
        _currentUtterance = -1;
        EmptyState.Visibility = Visibility.Visible;
    }

    private void OnSpeakerClicked(object sender, ItemClickEventArgs e)
    {
        if (e.ClickedItem is SpeakerRow row) _ = ShowSpeakerDialogAsync(row);
    }

    /// <summary>
    /// Name a speaker, mark them as the user, or merge two speakers. Merge exists because
    /// automatic labelling sometimes splits one person across two labels.
    /// </summary>
    private async Task ShowSpeakerDialogAsync(SpeakerRow row)
    {
        var nameBox = new TextBox
        {
            Header = "Name",
            PlaceholderText = "e.g. Priya",
            Text = row.Named ? row.Label.Replace(" (You)", string.Empty) : string.Empty,
        };

        var isSelf = new CheckBox { Content = "This is me", IsChecked = row.IsSelf };
        var hint = new TextBlock
        {
            Text = "Your own lines appear dimmed, with a clarity score you can read back.",
            TextWrapping = TextWrapping.Wrap,
            Opacity = 0.7,
        };

        var merge = new ComboBox
        {
            Header = "Same person as",
            HorizontalAlignment = HorizontalAlignment.Stretch,
        };
        merge.Items.Add("Nobody");
        var mergeIds = new List<int>();
        foreach (var other in Speakers)
        {
            if (other.Id == row.Id) continue;
            merge.Items.Add(other.Label);
            mergeIds.Add(other.Id);
        }
        merge.SelectedIndex = 0;

        var panel = new StackPanel { Spacing = 12, Width = 320 };
        panel.Children.Add(nameBox);
        panel.Children.Add(isSelf);
        panel.Children.Add(hint);
        panel.Children.Add(merge);

        var dialog = new ContentDialog
        {
            Title = "Edit speaker",
            Content = panel,
            PrimaryButtonText = "Save",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
            XamlRoot = Content.XamlRoot,
        };

        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;

        var name = nameBox.Text.Trim();
        if (!string.IsNullOrEmpty(name)) await _client.RenameSpeakerAsync(row.Id, name);
        await _client.SetSelfAsync(row.Id, isSelf.IsChecked == true);
        if (merge.SelectedIndex > 0)
            await _client.MergeSpeakersAsync(row.Id, mergeIds[merge.SelectedIndex - 1]);
    }

    // ---------- x:Bind helpers (static so compiled bindings can call them) ----------

    public static Visibility BoolToVisible(bool value) =>
        value ? Visibility.Visible : Visibility.Collapsed;

    /// <summary>
    /// Provisional text is dimmed; the user's own lines are dimmed further so they never
    /// compete with what other people said.
    /// </summary>
    public static double LineOpacity(bool isFinal, bool isSelf) =>
        isSelf ? (isFinal ? 0.55 : 0.4) : (isFinal ? 1.0 : 0.6);

    public static Brush SpeakerBrush(int index) =>
        (Brush)Application.Current.Resources[$"Speaker{index % 8}"];

    public static string ClarityText(int? clarity) => $"clarity {clarity ?? 0}%";

    public static Brush ClarityBrush(int? clarity)
    {
        var key = clarity switch
        {
            >= 80 => "ClarityGood",
            >= 55 => "ClarityMid",
            _ => "ClarityLow",
        };
        return (Brush)Application.Current.Resources[key];
    }
}
