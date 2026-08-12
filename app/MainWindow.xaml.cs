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
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Animation;
using Windows.Graphics;
using Windows.Security.Authorization.AppCapabilityAccess;

namespace Sunno;

/// <summary>A microphone the backend can capture from.</summary>
public sealed record AudioDevice(int Index, string Name, string HostApi, bool Loopback = false,
                                 bool IsDefault = false);

/// <summary>A model shown in first-run setup.</summary>
public sealed record ModelChoice(string Id, string Name, string Detail, int ApproxMb, bool Available,
                                 int LagMs = 0, bool Responsive = true)
{
    public string SizeLabel => ApproxMb >= 1024
        ? $"{ApproxMb / 1024.0:0.0} GB"
        : $"{ApproxMb} MB";

    /// <summary>
    /// Description prefixed with the expected delay.
    ///
    /// The slow ones no longer explain themselves inline. They used to read "(~5s delay,
    /// fine for video, too slow for conversation)", which repeats on every row what the
    /// group heading above them already says once, and asks a first-time user to work out
    /// which of two use cases they are. The number is the honest part; the heading carries
    /// the judgement.
    /// </summary>
    public string DetailWithSpeed
    {
        get
        {
            if (LagMs <= 0) return Detail;
            var delay = LagMs < 1000 ? $"~{LagMs / 1000.0:0.0}s" : $"~{LagMs / 1000.0:0}s";
            return $"({delay} delay) {Detail}";
        }
    }

    /// <summary>
    /// What a screen reader announces for this row.
    ///
    /// Records generate a ToString that dumps every property, and a ListView item with no
    /// AutomationProperties.Name falls back to it, so each option announced itself as
    /// "ModelChoice { Id = small, Name = Whisper small, Detail = ..., ApproxMb = 490,
    /// Available = True, LagMs = 777, Responsive = True, ... }". On the one screen where
    /// someone commits to a multi-gigabyte download, in an app built for people who rely on
    /// assistive technology, that is the difference between a choice and a wall of noise.
    /// </summary>
    public override string ToString()
    {
        var size = Available ? "already downloaded" : SizeLabel;
        return $"{Name}, {size}. {DetailWithSpeed}";
    }
}

public sealed partial class MainWindow : Window, System.ComponentModel.INotifyPropertyChanged
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

    /// <summary>
    /// Last reported model, kept for the diagnostics report.
    ///
    /// The compute device comes from the model catalogue frame, never from the status frame:
    /// that one's "device" is the *audio* device name (server/app.py sets it to
    /// stream.device_name), and reading it as a compute device once printed a hearing aid's name
    /// into a report that promised it held no device names.
    /// </summary>
    private string? _activeModel;
    private string? _computeDevice;
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

    /// <summary>
    /// Caption text size; the transcript <see cref="ItemsControl"/> binds its FontSize to this and
    /// the caption RichTextBlock inherits it. Observable so a change flows to every on-screen line
    /// without rebuilding the list. Seeded from the persisted setting, clamped in case the file is
    /// stale or corrupt.
    /// </summary>
    private double _captionSize = 26;
    public double CaptionSize
    {
        get => _captionSize;
        private set
        {
            if (_captionSize == value) return;
            _captionSize = value;
            PropertyChanged?.Invoke(this, new System.ComponentModel.PropertyChangedEventArgs(nameof(CaptionSize)));
        }
    }

    public event System.ComponentModel.PropertyChangedEventHandler? PropertyChanged;

    public ObservableCollection<CaptionLine> Lines { get; } = new();
    public ObservableCollection<SpeakerRow> Speakers { get; } = new();
    public ObservableCollection<ModelRow> Models { get; } = new();

    public MainWindow()
    {
        // Seed the persisted caption size before the XAML binding reads it. Clamped so a stale or
        // corrupt settings file can't push the transcript to an absurd size.
        _captionSize = Math.Clamp(_settings.CaptionFontSize, MinFont, MaxFont);
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
        // Subscribed and enqueued ahead of the roster it precedes. TryEnqueue preserves order
        // on the dispatcher queue, so the remap always runs before the relabel that depends
        // on it.
        _client.SpeakersMerged += (from, into) => _ui.TryEnqueue(() => OnSpeakersMerged(from, into));
        _client.SpeakerDeleted += (id, label) => _ui.TryEnqueue(() => OnSpeakerDeleted(id, label));
        _client.ConnectionChanged += ok => _ui.TryEnqueue(() => OnConnection(ok));
        _client.ModelRequired += (device, m) => _ui.TryEnqueue(() => OnModelRequired(device, m));
        _client.DownloadProgress += p => _ui.TryEnqueue(() => OnDownloadProgress(p));
        _client.DownloadComplete += _ => _ui.TryEnqueue(OnDownloadComplete);
        _client.DownloadFailed += msg => _ui.TryEnqueue(() => OnDownloadFailed(msg));
        _client.ModelCatalog += (current, device, list) =>
            _ui.TryEnqueue(() => OnModelCatalog(current, device, list));
        _backend.Crashed += (reason, detail) => _ui.TryEnqueue(() => OnBackendCrashed(reason, detail));

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

        // Windows does the asking. It raises its own consent prompt the first time the device
        // is actually opened - verified on a packaged install, where the per-app registry entry
        // flipped from Prompt to Allow at the moment capture started. That prompt is the
        // trustworthy one: it is native, the user has seen it before, and Windows records the
        // answer so nobody is asked twice.
        //
        // So "undecided" is treated as "go ahead and try". Only a recorded refusal keeps the
        // microphone shut, because Windows will not prompt again for those and the user has to
        // be sent to Settings instead.
        var micStatus = MicrophoneAccess.Check();
        _micStatus = micStatus;
        _lastGoodModel = _settings.Model;
        _micGranted = micStatus is null
                          or AppCapabilityAccessStatus.Allowed
                          or AppCapabilityAccessStatus.UserPromptRequired;
        _startedPaused = !_micGranted;

        var error = _backend.Start(
            device: _settings.DeviceIndex?.ToString(),
            model: _settings.Model,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused,
            loopbackDevice: _settings.LoopbackDeviceIndex,
            computeDevice: _settings.ForceCpu ? "cpu" : "auto");
        // Through the banner, not SetStatus. SetStatus writes to the small elapsed-time label in
        // the corner, which is sized for "1:08" - a failure sentence put there is invisible, so
        // an install missing its engine showed the loading text and nothing else, forever. The
        // two other Start/Restart call sites already report failures this way.
        if (!string.IsNullOrEmpty(error)) ShowFatalBackendError(error);
        App.Trace($"backend.Start -> {(string.IsNullOrEmpty(error) ? "ok" : error)}");
        _client.Start();
        _ = LoadDevicesAsync();
        App.Trace("MainWindow ctor: backend started");

        // Consent is asked from the content's Loaded event rather than window activation: a
        // window that is shown without being focused still needs to ask, otherwise the backend
        // sits paused behind a dialog that never appears.
        if (Content is FrameworkElement root)
        {
            RegisterShortcuts(root);

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
        // Before any caption can arrive, so the first line already agrees with the preference.
        Sunno.Models.CaptionLine.ClarityEnabled = _settings.ShowClarity;

        if (MicaController.IsSupported())
            SystemBackdrop = new MicaBackdrop { Kind = MicaKind.Base };

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);

        AppWindow.Resize(new SizeInt32(1040, 660));
        if (AppWindow.Presenter is OverlappedPresenter presenter)
        {
            // Honour the saved choice. Defaults to on, because captions need to stay readable
            // over whatever the user is actually doing, but someone who turned it off should
            // not find it back on at every launch.
            presenter.IsAlwaysOnTop = _settings.AlwaysOnTop;
            presenter.PreferredMinimumWidth = 720;
            presenter.PreferredMinimumHeight = 420;
        }
        AlwaysOnTopItem.IsChecked = _settings.AlwaysOnTop;
    }

    // ---------- caption stream ----------

    private void OnPartial(CaptionEvent ev)
    {
        if (string.IsNullOrWhiteSpace(ev.Text)) return;
        SetEmptyStateVisible(false);

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

        SetEmptyStateVisible(false);
        var line = _provisional is not null && _currentUtterance == ev.Id
            ? _provisional
            : AddLine(ev.Id);

        Apply(line, ev, isFinal: true);
        _provisional = null;
        _currentUtterance = -1;
        ScrollToEnd();
        AnnounceCaption(line);
    }

    /// <summary>
    /// Speak a finalised caption to a screen reader or braille display.
    ///
    /// AutomationProperties.LiveSetting on the transcript is metadata only: it tells an
    /// assistive client how urgently to treat a change, but nothing in WinUI watches the
    /// items and raises the event, so on its own it announces precisely nothing. The event
    /// has to be raised by hand.
    ///
    /// Announced on the final only, never on provisionals. A provisional is rewritten every
    /// few hundred milliseconds as the decoder revises itself, and speaking each revision
    /// would produce a stutter of half-sentences that contradict each other — the audible
    /// form of words appearing and disappearing that don't match what was said.
    ///
    /// ListenerExists is checked first so the peer tree isn't built when nothing is
    /// listening; creating peers has a real cost on a control that updates this often.
    /// </summary>
    private void AnnounceCaption(CaptionLine line)
    {
        try
        {
            if (!AutomationPeer.ListenerExists(AutomationEvents.LiveRegionChanged))
            {
                TraceAnnounceOnce("announce: no listener");
                return;
            }

            var text = line.Text;
            if (string.IsNullOrWhiteSpace(text)) return;

            // DisplayLabel, not SpeakerLabel — the same property the visible line and the
            // clipboard use. SpeakerLabel is the raw name, so a user's own line would be read
            // out under their name while the screen showed "You", and a self line whose raw
            // label is null would show "You:" but be announced with no prefix at all. Someone
            // relying on speech should hear the transcript the sighted user is reading.
            CaptionAnnouncer.Text = line.HasSpeaker ? $"{line.DisplayLabel}: {text}" : text;

            var peer = FrameworkElementAutomationPeer.CreatePeerForElement(CaptionAnnouncer);
            TraceAnnounceOnce($"announce: peer={(peer is null ? "null" : peer.GetType().Name)}");
            peer?.RaiseAutomationEvent(AutomationEvents.LiveRegionChanged);
        }
        catch (Exception ex)
        {
            // Announcement is an enhancement; never let it interrupt the visible transcript.
            TraceAnnounceOnce($"announce failed: {ex.GetType().Name}");
        }
    }

    private string? _lastAnnounceTrace;

    /// <summary>
    /// Trace the announcement path only when its outcome changes. This runs once per
    /// finalised utterance, and a file append per caption would mean thousands of writes
    /// across an afternoon of captioning to record the same answer over and over.
    /// </summary>
    private void TraceAnnounceOnce(string message)
    {
        if (_lastAnnounceTrace == message) return;
        _lastAnnounceTrace = message;
        App.Trace(message);
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

    /// <summary>
    /// Put the newest caption at the bottom of the view.
    ///
    /// The layout pass is forced first, deliberately. ScrollableHeight is derived from the
    /// extent as it currently stands, and every caller of this runs from the handler that has
    /// just added or replaced a line — so reading it straight away returns the height the list
    /// had *before* the new words existed, and the view stops a line short. The newest caption
    /// is the one the user is reading, and it was being clipped by the command bar.
    /// </summary>
    private void ScrollToEnd()
    {
        CaptionScroller.UpdateLayout();
        CaptionScroller.ChangeView(null, CaptionScroller.ScrollableHeight, null, disableAnimation: false);
    }

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
        // Frames that arrive after the engine has been declared dead are stale, and acting on
        // them is worse than ignoring them. The crash is reported from Process.Exited while
        // status frames come off the socket - two independent producers, so TryEnqueue's FIFO
        // guarantees nothing between them. A backend that reports a fault and then exits leaves
        // that last frame sitting in the socket buffer, which outlives the process: dispatched
        // after the fatal banner, an "error" frame repaints it and clears the crash detail, and
        // a "listening" frame closes the banner outright and puts "Listening for speech" back on
        // screen while nothing is being captured.
        //
        // Safe to drop only because the deliberate restart paths clear _backendFatal before they
        // revive the engine; without that this would ignore the recovered backend's own frames
        // and leave a permanent "speech engine stopped" over a working app.
        if (_backendFatal) return;

        if (st.Running is bool running) SetRunning(running);
        _backendLoading = st.State == "loading";

        // Remember the model the engine reported, for the diagnostics report. Only overwrite on
        // a frame that actually carries it: error frames and plain running/paused updates leave
        // it null, and losing it would make the report say "unknown" for the rest of the session.
        // st.Device is deliberately not captured; see the _activeModel declaration.
        if (!string.IsNullOrEmpty(st.Model)) _activeModel = st.Model;

        if (st.State == "error")
        {
            ShowActionableError(st);
            // The device failed to open, so the switch is over — it just ended badly. The
            // backend does send "stopped" a frame later, which would clear this anyway, but
            // one frame of "microphone unavailable" beside a spinner saying the microphone is
            // coming up is the exact mixed message this indicator exists to remove.
            SetDeviceBusy(false);
            // And stop the centre promising the same thing. ShowActionableError sets
            // _micProblem, which makes the "stopped" frame arriving next return early to keep
            // the real reason on screen — so nothing downstream ever retires the loading panel,
            // and a failed switch would leave "Switching to <device>" spinning indefinitely.
            // That is the failure ShowFailedState was written for; its text defers to the
            // InfoBar this branch just raised.
            ShowFailedState();
            // Both cases have a full explanation elsewhere: the InfoBar names the problem and
        // offers the fix, and the centre state describes what the app is doing. Repeating a
        // two-word summary in the corner adds nothing.
        ClearStatus();
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
            // The same signal that commits a model switch ends a device switch: the engine has
            // finished loading and the pipeline is up. "stopped" counts — a switch onto a device
            // while capture is paused still completed, and waiting for "listening" would leave
            // the ring turning until the user pressed play.
            SetDeviceBusy(false);
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

        // The centre of the window already carries loading, paused and error states in full,
        // and failures get an InfoBar. Anything repeated down here is noise beside the
        // recording clock, so those cases blank the line instead.
        if (st.State == "loading") { ClearStatus(); return; }

        SetStatus(st.State switch
        {
            "stopped" => "Paused",
            _ => st.State,
        });
    }

    /// <summary>
    /// Blank the status line. Used where the state is already shown more prominently
    /// elsewhere — the centre empty state or the InfoBar — so this corner stays what it
    /// looks like: a recording indicator, not a status bar.
    /// </summary>
    private void ClearStatus()
    {
        _elapsedTimer.Stop();
        ToolTipService.SetToolTip(StatusText, _captureClock.Elapsed > TimeSpan.Zero
            ? $"{FormatElapsed(_captureClock.Elapsed)} recorded in this conversation"
            : null);
        StatusText.Text = string.Empty;
    }

    /// <summary>
    /// Hand the status line to the conversation timer. The device name moves to the picker;
    /// how long this conversation has been running is the more useful thing to show, and it
    /// is the one piece of state that says the microphone is open right now.
    /// </summary>
    private void ShowElapsed()
    {
        // Start, not Restart. The clock measures the conversation, not one uninterrupted
        // stretch of it, so pausing for an aside and resuming continues the count rather
        // than throwing away everything before the pause. Stopwatch accumulates across
        // Stop/Start, which is exactly this behaviour; Start on a running clock is a no-op,
        // so repeated "listening" reports during one run are harmless.
        _captureClock.Start();
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

    private const string RecordingTimeHint = "How long this conversation has been recording";
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

        // Applies to loopback as well as the microphone. The loopback stream synthesises
        // silence while an output endpoint is merely idle, so levels keep flowing and a
        // paused video no longer trips this. Levels stop only when capture is genuinely
        // dead — including an output device that disappears mid-session, which otherwise
        // leaves a running clock above a transcript that will never gain another line.
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
    /// Stops the ticking, never the clock. Losing the socket does not stop the microphone,
    /// and even where capture really does stop the elapsed time is kept: it measures the
    /// conversation, which outlives any one capture run.
    ///
    /// The accumulated time moves into the tooltip rather than vanishing. The line itself
    /// stays with the message — while paused that message is "microphone released", which is
    /// a privacy assertion and outranks a number the user can still hover for.
    /// </summary>
    private void SetStatus(string text)
    {
        _elapsedTimer.Stop();
        // The microphone really is released while paused, and Windows' own indicator says so
        // — but it belongs on hover rather than on the line, which stays short.
        ToolTipService.SetToolTip(StatusText, _captureClock.Elapsed > TimeSpan.Zero
            ? $"{FormatElapsed(_captureClock.Elapsed)} recorded · microphone released"
            : "Microphone released");
        StatusText.Text = text;
    }

    /// <summary>
    /// Capture stopped: freeze the count where it is.
    ///
    /// Deliberately not a reset. Ducking out of a conversation for a private aside is the
    /// whole point of the pause button, and zeroing a forty-minute reading because someone
    /// stepped away for thirty seconds would punish exactly the behaviour the control exists
    /// to encourage. Resuming continues from here; only a new conversation starts over.
    /// </summary>
    private void PauseCaptureClock()
    {
        _elapsedTimer.Stop();
        _captureClock.Stop();
        _sinceLevel.Reset();
    }

    /// <summary>A new conversation: the transcript was cleared, so the count starts over.</summary>
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
        // No longer describing a crash, so the copy-details payload must not survive: OnMicAction
        // takes the copy branch whenever _crashDetail is non-null, which meant a microphone
        // banner offering "Open Settings" quietly copied stale crash text instead.
        _crashDetail = null;
        // Taking the bar over, so drop any stickiness it inherited. Otherwise a device notice
        // raised seconds earlier keeps this bar pinned, and OnStatus will not close it once the
        // user has picked a working microphone: they would read "Microphone unavailable" while
        // captions were flowing. ShowFatalBackendError does the same for the same reason.
        _infoSticky = false;

        switch (st.Code)
        {
            case "mic_denied":
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

            case "capture_failed":
                MicInfoBar.Severity = InfoBarSeverity.Warning;
                MicInfoBar.Title = "Microphone unavailable";
                MicInfoBar.Message =
                    (st.Message ?? "The microphone could not be opened.") +
                    " Try choosing a different microphone below.";
                MicActionLink.Visibility = Visibility.Collapsed;
                break;

            default:
                // Never the backend's own words. An unrecognised failure used to print
                // st.Message straight onto the banner, which is how "[Errno -9996] Invalid
                // device info" ended up as the entire explanation offered to the user.
                MicInfoBar.Severity = InfoBarSeverity.Error;
                MicInfoBar.Title = "Something went wrong";
                MicInfoBar.Message =
                    "Sunno hit a problem it doesn't have a specific explanation for. "
                    + "Restarting Sunno usually clears it.";
                // Details come from the engine's log through IsDiagnostic, not from st.Message.
                // _crashDetail is embedded verbatim in the diagnostics export, under a header
                // promising no transcript text, no speaker names and no device names - and an
                // arbitrary backend-supplied string is exactly what would make that promise
                // unenforceable. The same narrowing the crash path uses applies here.
                var engineDetail = _backend.RecentDiagnostics();
                _crashDetail = string.IsNullOrWhiteSpace(engineDetail)
                    ? $"No further detail was recorded.\n\nLog: {BackendHost.DisplayLogPath}"
                    : $"{engineDetail}\n\nLog: {BackendHost.DisplayLogPath}";
                MicActionLink.Content = "Copy details";
                MicActionLink.Visibility = Visibility.Visible;
                break;
        }
        MicInfoBar.IsOpen = true;
    }

    /// <summary>
    /// The backend died. Say so plainly and point at the log — a user who is relying on this to
    /// follow a conversation must never be left watching a spinner that will never resolve.
    /// </summary>
    private void OnBackendCrashed(string reason, string detail)
    {
        App.Trace($"backend crashed: {reason}");
        _backendLoading = false;
        // Dropped here rather than beside the fatal banner below, because two branches of this
        // method start a fallback model and return without ever reaching it. A crash during a
        // device switch takes one of them — the device path clears _engineReadyThisSession,
        // which is what the fallback branch keys off — so the ring would otherwise keep turning
        // through a thirty second engine rebuild, still claiming to be changing microphone.
        SetDeviceBusy(false);

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
        ShowFatalBackendError(reason, detail);
    }

    /// <summary>
    /// Report a backend we cannot recover from. Split out so the restart path can reuse it:
    /// a failed restart has already killed the old process, so silently returning would leave
    /// "Reconnecting…" on screen forever with nothing left to reconnect to.
    ///
    /// The banner gets the human reason; the technical detail goes to "Copy details" only. It
    /// used to carry both, so a port conflict presented the user with a Python traceback full of
    /// absolute paths as the whole explanation.
    /// </summary>
    private void ShowFatalBackendError(string reason, string? detail = null)
    {
        _backendFatal = true;
        _backendLoading = false;
        // The engine is dead, so nothing is coming up. A device ring still turning here would
        // promise a microphone that is never going to arrive, next to a banner saying the
        // opposite.
        SetDeviceBusy(false);
        ClearStatus();
        _micProblem = false;
        _infoSticky = false;
        MicInfoBar.Severity = InfoBarSeverity.Error;
        MicInfoBar.Title = "Sunno's speech engine stopped";
        MicInfoBar.Message = reason;
        MicActionLink.Content = "Copy details";
        MicActionLink.Visibility = Visibility.Visible;
        MicInfoBar.IsOpen = true;
        ShowFailedState();

        _crashDetail = string.IsNullOrWhiteSpace(detail)
            ? $"{reason}\n\nLog: {BackendHost.DisplayLogPath}"
            : $"{reason}\n\n{detail}\n\nLog: {BackendHost.DisplayLogPath}";
    }

    /// <summary>
    /// The last fatal backend detail, for "Copy details".
    ///
    /// Cleared whenever the bar stops describing a crash. It was previously set once and never
    /// reset, so after any fatal error a later microphone problem rendered a link labelled
    /// "Open Settings" that copied stale crash text instead of doing anything useful.
    /// </summary>
    private string? _crashDetail;

    /// <summary>
    /// Take the engine out of its dead state, for the paths that deliberately start a new one.
    ///
    /// _backendFatal is otherwise set-once: it stops reconnect chatter painting over a real
    /// failure. But a model switch, a device change and the Force CPU toggle all replace the
    /// process on purpose, and leaving the flag set makes the app ignore the new process's own
    /// status frames while permanently suppressing device notices. Also clears the crash detail,
    /// which describes a process that is no longer the one running.
    /// </summary>
    private void ClearBackendFatal()
    {
        _backendFatal = false;
        _crashDetail = null;
    }

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
        // Every remaining state is one Windows has already recorded a refusal for, and it will
        // not prompt again for those - so the only useful action is the Settings app. There is
        // no "ask again" branch any more: undecided consent never reaches this bar, because
        // Windows raises its own prompt when the device is opened.
        await Windows.System.Launcher.LaunchUriAsync(
            new Uri("ms-settings:privacy-microphone"));
    }

    /// <summary>
    /// Settle microphone consent.
    ///
    /// The app no longer asks. Windows raises its own prompt the first time the microphone is
    /// actually opened — verified on a packaged install, where the per-app consent entry went
    /// from Prompt to Allow at the moment capture started. Asking first meant asking a question
    /// the OS was about to ask properly: two dialogs for one decision, and only one of them
    /// recorded anywhere.
    ///
    /// What is left here is reporting a refusal Windows has already recorded, because those
    /// states never produce a prompt and the only way out is the Settings app.
    /// </summary>
    private Task EnsureMicrophoneAccessAsync()
    {
        // Deliberately reuses the status read during construction rather than calling
        // CheckAccess again. A second call from inside the Loaded handler reproducibly took
        // the process down with a stowed exception in Microsoft.UI.Xaml, and re-reading buys
        // nothing: consent cannot change in the ~100 ms between the two, and AccessChanged
        // covers any change afterwards.
        var status = _micStatus;
        App.Trace($"mic: reusing status={status}");

        ApplyMicrophoneStatus(status);

        // Notice a later grant from Settings, so the fallback isn't a dead end that needs a relaunch.
        MicrophoneAccess.Changed += OnMicAccessChanged;
        App.Trace("mic: done");
        return Task.CompletedTask;
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
        //
        // UserPromptRequired belongs here too, and that is the point of letting Windows ask:
        // it means the OS has no answer on file and will raise its own prompt the moment the
        // device is opened. Showing a banner for it put a question in front of the user that
        // Windows was about to ask properly, and the banner's own button was the thing that
        // had no way to answer it.
        if (status is null
                   or AppCapabilityAccessStatus.Allowed
                   or AppCapabilityAccessStatus.UserPromptRequired)
        {
            // An explicit "not now" outranks the OS default, which for a full-trust app is
            // always Allowed and would otherwise silently overturn the user's own answer.
            if (_micDeclined) return;

            _micProblem = false;
            // Only the microphone's own message. This runs on every launch where access is
            // fine, and it used to close the bar outright, which discarded any sticky notice
            // raised moments earlier — the device-rot warning always, and a model fallback
            // notice whenever one arrived before consent resolved. Same idiom as OnStatus.
            //
            // _backendFatal is checked for the same reason: a start failure is reported from the
            // constructor, which runs before this, so closing the bar here erased the only
            // explanation on screen and left the empty state pointing at a banner that was no
            // longer there.
            //
            // This guard is load-bearing and is NOT the redundant one. It protects a *close*,
            // and there is no central guard for closing the bar. The qualifier on the
            // RenderDeviceNotice() call below is the redundant one - RenderDeviceNotice guards
            // itself. If either is ever trimmed, trim that one, never this.
            if (!_infoSticky && !_backendFatal) MicInfoBar.IsOpen = false;
            // Put back a device notice that a permission problem had taken the bar over from —
            // but never over a fatal one. "Your microphone changed" is an informational notice;
            // painting it over "the speech engine stopped" would replace the only explanation
            // of why nothing is being captioned with a message implying everything is fine.
            if (!_backendFatal) RenderDeviceNotice();
            _micGranted = true;
            TryStartCapture();
            return;
        }

        _micProblem = true;
        // A dead engine outranks a microphone problem, and the state above is still worth
        // recording — the bar just must not be repainted. Two things go wrong otherwise: the
        // banner blames the microphone for an engine crash and erases the real explanation,
        // and the link below is relabelled "Open Settings" while _crashDetail is still set,
        // which makes OnMicAction take its copy branch and silently put a crash dump on the
        // clipboard instead of opening anything.
        //
        // This is the second writer to reach the bar without knowing about _backendFatal (the
        // first was OnStatus). If a third appears, the guards should become one ShowInfoBar
        // chokepoint that refuses to overwrite a fatal banner, rather than a rule re-stated at
        // every writer.
        if (_backendFatal) return;
        // Taking the bar over for a microphone problem, so release any sticky notice first.
        // Otherwise the check in the recovery path above never fires and the "microphone access
        // is off, nothing can be transcribed" banner stays on screen after the user has turned
        // access back on: they follow the app's instructions, it appears to do nothing, and a
        // deaf user is told captioning is dead while it is running.
        _infoSticky = false;
        MicInfoBar.Severity = InfoBarSeverity.Warning;
        MicActionLink.Visibility = Visibility.Visible;

        switch (status)
        {
            case AppCapabilityAccessStatus.DeniedByUser:
                MicInfoBar.Title = "Microphone access is off";
                MicInfoBar.Message =
                    "Microphone access for Sunno is turned off, so nothing can be " +
                    "transcribed. Turn it back on under Privacy & security › Microphone.";
                break;

            case AppCapabilityAccessStatus.DeniedBySystem:
                MicInfoBar.Title = "Microphone is off for this device";
                MicInfoBar.Message =
                    "Microphone access is turned off for the whole device, or for all desktop " +
                    "apps, so no app can transcribe. Turn it on under Privacy & security › " +
                    "Microphone.";
                break;

            default:
                // NotDeclaredByApp: a packaging defect, not something the user can fix.
                MicInfoBar.Severity = InfoBarSeverity.Error;
                MicInfoBar.Title = "Microphone capability missing";
                MicInfoBar.Message =
                    "This build didn't declare the microphone capability, so Windows won't " +
                    "grant access. Reinstalling from a complete package should fix it.";
                MicActionLink.Visibility = Visibility.Collapsed;
                break;
        }

        // Always Settings now. Every state that reaches here is a refusal Windows has recorded
        // and will not re-prompt for, so there is nothing this app can ask.
        MicActionLink.Content = "Open Settings";
        MicInfoBar.IsOpen = true;

        // Retire the centre panel too, or it goes on promising captions that cannot arrive.
        //
        // Windows has already refused, so the backend is launched with startStopped and reports
        // "stopped" rather than an error — and _micProblem, set above, makes OnStatus swallow
        // that frame to keep this banner on screen. Nothing downstream then retires the loading
        // panel, so a first run with the microphone blocked sat on "Starting the speech engine,
        // this takes about half a minute the first time" indefinitely, beside a banner saying
        // access was off. The one state a deaf user cannot afford to misread is whether the app
        // is still coming up or has given up.
        //
        // Deliberately not paired with SetEmptyStateVisible: this rewrites the panel's contents
        // without forcing it over a transcript that is already on screen.
        ShowFailedState();
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
        // Microphone consent gates the microphone, not captioning.
        //
        // System audio is captured from an output endpoint, which Windows does not put behind
        // the microphone permission and which records nobody in the room. Gating it here meant
        // that someone whose microphone Windows has refused could not caption a video call
        // either, and was told nothing, because the refusal happens before anything draws a
        // banner. That is a reasonable state to be in, and the person most careful about a
        // microphone is exactly the one who ends up there.
        //
        // The real gate is _micGranted, which is false for every status Windows records as a
        // refusal. _micDeclined is carried along for symmetry with the two other places that
        // test the pair, but nothing assigns it true any more, so it contributes nothing here.
        var loopback = _settings.LoopbackDeviceIndex is not null;
        if (!loopback && (!_micGranted || _micDeclined)) return;
        if (!_startedPaused || _captureRequested || !_connected) return;
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

    /// <summary>
    /// The user closed the bar. Drop the stickiness and forget the device notice: re-opening
    /// something they have just dismissed, which is what would happen the next time the
    /// microphone permission path re-rendered it, is its own bug.
    /// </summary>
    private void OnMicInfoClosed(InfoBar sender, object args)
    {
        _infoSticky = false;
        _deviceNotice = null;
    }

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

        // Re-ask for the catalogue on open. The delay figures are learned from real decodes,
        // and the first fetch happens on connect before a single utterance has been timed —
        // so without this the user would keep seeing the shipped estimate all session and
        // only get their own machine's number after a restart.
        if (_modelSectionOpen && _connected) _ = _client.RequestModelsAsync();

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

    /// <summary>
    /// Whether the user wants animation at all.
    ///
    /// Settings > Accessibility > Visual effects > Animation effects. Sunno exists for people
    /// with a disability, and motion sensitivity is one; someone who has turned animation off
    /// system-wide has asked every app to stop, and an accessibility app has less excuse than
    /// most to ignore that. Read per use rather than cached, because the setting can change
    /// while the app is running.
    /// </summary>
    private static bool AnimationsWanted
    {
        get
        {
            try { return new Windows.UI.ViewManagement.UISettings().AnimationsEnabled; }
            catch { return true; }   // never let a diagnostics lookup break the UI
        }
    }

    // Held so they can be stopped before a direct assignment. A completed Storyboard defaults to
    // FillBehavior.HoldEnd, and an animation value outranks a local value in dependency-property
    // precedence, so assigning Height or Angle directly while a previous run still holds the
    // property is silently ignored. Reachable by animating with effects on, then turning
    // "Animation effects" off — which AnimationsWanted is written to observe live — and then
    // collapsing the panel, which would otherwise refuse to close.
    private Storyboard? _modelPanelStory;
    private Storyboard? _chevronStory;

    private void AnimateModelPanel(double toHeight)
    {
        _modelPanelStory?.Stop();

        if (!AnimationsWanted)
        {
            ModelPanel.Height = toHeight;
            return;
        }

        var animation = new DoubleAnimation
        {
            To = toHeight,
            Duration = new Duration(TimeSpan.FromMilliseconds(220)),
            EnableDependentAnimation = true,
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut },
        };
        Storyboard.SetTarget(animation, ModelPanel);
        Storyboard.SetTargetProperty(animation, "Height");

        _modelPanelStory = new Storyboard();
        _modelPanelStory.Children.Add(animation);
        _modelPanelStory.Begin();
    }

    private void AnimateChevron(double angle)
    {
        _chevronStory?.Stop();

        if (!AnimationsWanted)
        {
            ModelChevronRotate.Angle = angle;
            return;
        }

        var animation = new DoubleAnimation
        {
            To = angle,
            Duration = new Duration(TimeSpan.FromMilliseconds(220)),
            EnableDependentAnimation = true,
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut },
        };
        Storyboard.SetTarget(animation, ModelChevronRotate);
        Storyboard.SetTargetProperty(animation, "Angle");

        _chevronStory = new Storyboard();
        _chevronStory.Children.Add(animation);
        _chevronStory.Begin();
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

    private void OnModelCatalog(string current, string? computeDevice,
                                IReadOnlyList<ModelOption> options)
    {
        // "cuda" or "cpu", for the diagnostics report. This frame is the only one that carries
        // it; the status frame's "device" is the audio device name.
        if (!string.IsNullOrEmpty(computeDevice)) _computeDevice = computeDevice;

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
                    LagMs = o.LagMs,
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
        // Deliberately reviving the engine, so it is no longer dead. Left set, this would
        // outlive the failure it described: the new process's own status frames are ignored
        // while _backendFatal holds, and the device-rot notice stays suppressed for the rest
        // of the session. Cleared here rather than on the frame that proves recovery, because
        // that frame is exactly what the flag would block.
        ClearBackendFatal();
        // The backend is about to be replaced, so the microphone closes — but the transcript
        // survives and so does the conversation, so the count freezes rather than resetting.
        PauseCaptureClock();

        // Deliberately NOT persisted yet. A model that downloads but fails to load would
        // otherwise become the choice reloaded on every future launch, turning one bad switch
        // into a crash loop with no way out from inside the app.
        var error = _backend.Restart(
            device: _settings.DeviceIndex?.ToString(),
            model: id,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused,
            loopbackDevice: _settings.LoopbackDeviceIndex,
            computeDevice: _settings.ForceCpu ? "cpu" : "auto");

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
        LoadingRing.IsAnimating = true;
        LoadingRing.Visibility = Visibility.Visible;
        EmptyGlyph.Visibility = Visibility.Collapsed;
        EmptyTitle.Text = title;
        EmptyDetail.Text = "This takes about half a minute.";
        SetEmptyStateVisible(Lines.Count == 0);
    }

    /// <summary>
    /// Show or hide the empty state, crossfading rather than snapping.
    ///
    /// Every caller goes through here so the fade cannot be half-applied: the transcript
    /// arriving, the engine reloading and the transcript being cleared all toggle this, and a
    /// caption landing mid-fade must not leave a ghosted panel over the text.
    ///
    /// Opacity only. This panel is a sibling of CaptionScroller, not inside it, but opacity is
    /// composition-driven and costs no layout pass either way, which matters at exactly the
    /// moment this fires: the first word of a conversation arriving.
    /// </summary>
    private void SetEmptyStateVisible(bool visible)
    {
        if (_emptyStateVisible == visible) return;
        _emptyStateVisible = visible;

        _emptyFadeStory?.Stop();   // see the note on _modelPanelStory about held animation values

        if (!AnimationsWanted)
        {
            EmptyState.Opacity = visible ? 1 : 0;
            EmptyState.Visibility = visible ? Visibility.Visible : Visibility.Collapsed;
            return;
        }

        if (visible) EmptyState.Visibility = Visibility.Visible;

        var fade = new DoubleAnimation
        {
            To = visible ? 1 : 0,
            Duration = new Duration(TimeSpan.FromMilliseconds(160)),
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut },
        };
        Storyboard.SetTarget(fade, EmptyState);
        Storyboard.SetTargetProperty(fade, "Opacity");

        _emptyFadeStory = new Storyboard();
        _emptyFadeStory.Children.Add(fade);
        // Re-check the field rather than the captured value: the state can flip back during the
        // 160 ms, and collapsing on a stale decision would hide a panel that should be showing.
        _emptyFadeStory.Completed += (_, _) =>
        {
            if (!_emptyStateVisible) EmptyState.Visibility = Visibility.Collapsed;
        };
        _emptyFadeStory.Begin();
    }

    private bool _emptyStateVisible = true;
    private Storyboard? _emptyFadeStory;

    private void ShowReadyState()
    {
        LoadingRing.IsAnimating = false;
        LoadingRing.Visibility = Visibility.Collapsed;
        EmptyGlyph.Visibility = Visibility.Visible;
        EmptyTitle.Text = "Listening for speech";
        EmptyDetail.Text = "Captions appear here as people talk.";
    }

    /// <summary>Engine is up, but capture is paused — by the user, or awaiting consent.</summary>
    private void ShowIdleState()
    {
        LoadingRing.IsAnimating = false;
        LoadingRing.Visibility = Visibility.Collapsed;
        EmptyGlyph.Visibility = Visibility.Visible;
        EmptyTitle.Text = "Paused";
        // "Start", not "Resume": on first run, before consent, nothing has ever started.
        EmptyDetail.Text = "Press the microphone button to start listening.";
    }

    /// <summary>
    /// The engine is not coming up. Without this the centre of the window kept its loading
    /// text, so a failed start left "Starting the speech engine - this takes about half a
    /// minute the first time" on screen indefinitely: a promise of captions that will never
    /// arrive, which is the exact failure this app cannot afford.
    /// </summary>
    private void ShowFailedState()
    {
        LoadingRing.IsAnimating = false;
        LoadingRing.Visibility = Visibility.Collapsed;
        EmptyGlyph.Visibility = Visibility.Visible;
        EmptyTitle.Text = "Captions aren't running";
        EmptyDetail.Text = "The message above explains what happened.";
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
            if (DevicePicker.Items.Count == 0)
            {
                // Same idea for the microphone list, and for a sharper reason: the device list
                // is fetched over HTTP once from the constructor, and it gives up after twenty
                // seconds. A backend that fails to start - a port conflict, say - outlasts that,
                // so the picker ends up empty. Recovering the engine afterwards brought captions
                // back but left the user unable to change microphone for the rest of the
                // session, with an empty dropdown and no explanation.
                _ = LoadDevicesAsync();
            }
            return;
        }
        // A dead backend also looks "disconnected", and its reconnect attempts would otherwise
        // paint over the real explanation with a reassuring one.
        if (_backendFatal) return;
        // On a cold start the socket isn't up yet because the model is still loading.
        // "Starting…" is more truthful than "Reconnecting…" for a first run.
        // Startup and reconnection are both already on screen in the centre, with a spinner
        // and the "about half a minute" expectation. Saying it twice was the duplication
        // that made this corner look like a status bar.
        ClearStatus();
    }

    private void OnRoster(IReadOnlyList<SpeakerInfo> speakers)
    {
        // Reconciled in place rather than cleared and rebuilt.
        //
        // The roster is re-sent for any change to any speaker: a rename, marking someone as
        // yourself, a merge, a reset, or simply a new person being heard. Clearing the
        // collection recreated every row each time, which threw away the scroll position and
        // made the list-entrance animation fire on all four rows every time one person was
        // renamed. A row should animate when that person is genuinely new, and stay still
        // otherwise.
        //
        // Depends on the backend emitting speakers in ascending id order (server/speaker.py
        // sorts the roster), so a genuinely new person always arrives at the tail. Ids are
        // durable and are never reused, so they can have gaps in them: a merged-away speaker
        // leaves one behind. That is deliberate. Ids used to be list positions, which meant a
        // merge renumbered everyone above the merged-away person and the relabel pass below
        // then rewrote their already-scrolled captions with the next person's name.
        for (var i = Speakers.Count - 1; i >= 0; i--)
            if (!speakers.Any(s => s.Id == Speakers[i].Id))
                Speakers.RemoveAt(i);

        for (var i = 0; i < speakers.Count; i++)
        {
            var s = speakers[i];
            var label = s.IsSelf ? $"{s.Label} (You)" : s.Label;
            var existing = Speakers.FirstOrDefault(r => r.Id == s.Id);

            if (existing is null)
            {
                // The clamp is a guard against a malformed roster repeating an id, not routine:
                // with a well-formed one every entry processed so far is already in the
                // collection, so Count >= i always holds.
                Speakers.Insert(Math.Min(i, Speakers.Count), new SpeakerRow
                {
                    Id = s.Id,
                    Label = label,
                    IsSelf = s.IsSelf,
                    Named = s.Named,
                });
                continue;
            }

            // Assign through the properties so only what actually changed raises a notification.
            existing.Label = label;
            existing.IsSelf = s.IsSelf;
            existing.Named = s.Named;
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

    /// <summary>
    /// Two speakers turned out to be one person, so move the absorbed speaker's captions onto
    /// the survivor.
    ///
    /// This has to happen before the roster frame that follows: ids are durable, so the
    /// absorbed id is simply gone from the new roster, and OnRoster's relabel pass skips any
    /// line whose id it cannot find. Without this the merged-away person's lines would keep
    /// their old name permanently, which is the whole point of merging two speakers.
    /// </summary>
    private void OnSpeakersMerged(int from, int into)
    {
        foreach (var line in Lines)
            if (line.SpeakerId == from)
                line.SpeakerId = into;
    }

    /// <summary>
    /// A speaker was forgotten, so their captions stop claiming to know who spoke.
    ///
    /// The id is deliberately left on the line: it is retired and can never be handed to
    /// anyone else, so it keeps the lines grouped and their colour stable while no longer
    /// naming a person. IsSelf is cleared with it - if the deleted profile was the user's
    /// own, those lines should not go on being rendered as theirs.
    /// </summary>
    private void OnSpeakerDeleted(int id, string fallbackLabel)
    {
        foreach (var line in Lines)
        {
            if (line.SpeakerId != id) continue;
            line.SpeakerLabel = fallbackLabel;
            line.IsSelf = false;
        }
    }

    /// <summary>
    /// Forget a speaker, after checking the user means it when there is something to lose.
    ///
    /// Named speakers are confirmed because naming one pins their voice profile and that
    /// survives restarts: deleting is the only way to lose it, and it cannot be undone.
    /// Automatically discovered speakers are not confirmed - they carry no work of the
    /// user's and reappear the moment that person speaks again.
    /// </summary>
    private async Task DeleteSpeakerAsync(SpeakerRow row)
    {
        if (row.Named)
        {
            var name = row.Label.Replace(" (You)", string.Empty);
            var confirm = new ContentDialog
            {
                Title = $"Forget {name}?",
                Content = "Sunno will stop recognising this voice, and lines already in the "
                          + "transcript will no longer show their name. This cannot be undone.",
                PrimaryButtonText = "Forget",
                CloseButtonText = "Cancel",
                DefaultButton = ContentDialogButton.Close,
                XamlRoot = Content.XamlRoot,
            };
            try
            {
                if (await confirm.ShowAsync() != ContentDialogResult.Primary) return;
            }
            catch (Exception ex)
            {
                // One ContentDialog at a time per window; if another is somehow up, do
                // nothing rather than delete without having asked.
                App.Trace($"delete confirm failed to open: {ex.GetType().Name}");
                return;
            }
        }

        await _client.DeleteSpeakerAsync(row.Id);
    }

    private void SetRunning(bool running)
    {
        _running = running;
        ToggleGlyph.Glyph = running ? "\uE769" : "\uE720";   // pause bars / microphone
        ToggleButton.SetValue(AutomationProperties.NameProperty,
            running ? "Pause transcribing and release the microphone" : "Start transcribing");
        ToolTipService.SetToolTip(ToggleButton,
            running ? "Pause transcribing (Space)" : "Start transcribing (Space)");

        if (!running)
        {
            // Capture stopped; the status message that follows owns the line. The count is
            // frozen, not cleared — resuming continues the same conversation.
            PauseCaptureClock();
            LevelFill.Height = 0;
            if (_provisional is not null) Lines.Remove(_provisional);
            _provisional = null;
            _currentUtterance = -1;
        }
    }

    // ---------- first-run setup ----------

    private void OnModelRequired(string? computeDevice, IReadOnlyList<ModelOption> options)
    {
        // Recorded for the diagnostics report, which on a first run had no other source for
        // it: this is the only frame a machine with no model downloaded ever sees, and it was
        // dropping the field. The wording on this screen deliberately does not use it, because
        // naming the part of the PC that does the work does not help anyone choose.
        if (!string.IsNullOrEmpty(computeDevice)) _computeDevice = computeDevice;

        var choices = options
            .Select(o => new ModelChoice(o.Id, o.Name, o.Detail, o.ApproxMb, o.Available,
                                         o.LagMs, o.Responsive))
            .ToList();

        BuildModelGroups(choices);

        // Preselect the model this hardware can actually keep up with, preferring one
        // already on disk. Picking purely by "already downloaded" would start a CPU-only
        // machine on whatever happened to be cached, which may be the slowest option.
        //
        // The middle step deliberately differs from the backend, which does not look at disk
        // state at all (server/hardware.py default_model). When nothing keeps up but
        // something is already downloaded, this offers the downloaded one and the backend
        // would not. That is a kindness about a multi-gigabyte download, not an oversight.
        var preferred = choices
            .Where(m => m.Responsive)
            .OrderByDescending(m => m.Available)
            .FirstOrDefault()
            ?? choices.FirstOrDefault(m => m.Available)
            // Last resort: the fastest, matching server/hardware.py's
            // `min(catalog_ids, key=estimated_lag_ms)`. This used to be the first entry in
            // the catalogue, which is ordered most-accurate-first — so on a machine too slow
            // for anything, the two sides disagreed and the screen preselected the largest
            // download and the longest delay, which is the worst answer available.
            ?? choices.OrderBy(m => m.LagMs <= 0 ? int.MaxValue : m.LagMs).FirstOrDefault();
        ModelList.SelectedItem = preferred;

        SetupError.IsOpen = false;
        DownloadPanel.Visibility = Visibility.Collapsed;
        DownloadButton.IsEnabled = true;
        SetupOverlay.Visibility = Visibility.Visible;
        // The setup overlay fills the window; nothing needs restating underneath it.
        ClearStatus();
    }

    /// <summary>
    /// Sort the options into what this machine can keep up with and what it cannot.
    ///
    /// The delay was already written into each row, but a number in a description is easy to
    /// skim past when the thing next to it says "most accurate". Someone choosing a model has
    /// no way to know that 5 seconds is disqualifying for a conversation and fine for a film,
    /// so the split says it for them, and says it about *their* PC rather than in the abstract.
    ///
    /// Two headers rather than a filter: everything stays choosable. A machine that struggles
    /// today may be plugged into power tomorrow, and hiding options from someone who has
    /// already read the delay would be deciding for them.
    /// </summary>
    private void BuildModelGroups(List<ModelChoice> choices)
    {
        ModelList.Items.Clear();
        // No catalogue at all. Not reachable from today's backend, but the branch below would
        // otherwise print "None of these can keep up" over an empty list, which reads as a
        // verdict on models that were never offered.
        if (choices.Count == 0) return;

        var keepsUp = choices.Where(m => m.Responsive).ToList();
        var lags = choices.Where(m => !m.Responsive).ToList();

        // Deliberately no mention of processors or graphics cards. An earlier version said
        // "Recommended for your processor", which asks someone who has just installed a
        // captioning app to know which part of their PC does the work, and to know that the
        // answer changes the advice. "This PC" is the only part they need.
        //
        // The slower group also used to explain itself as "too slow for conversation, but
        // fine for video", which asks a first-time user to hold two use cases in their head
        // and decide which one they are. The consequence is the useful part: the words turn
        // up late. Say that, and let them judge.

        // Every option is too slow. A "not recommended" heading over the entire list would be
        // true and useless: it reads as "do not use this app". Say it once, plainly, and let
        // fastest-first carry the advice.
        if (keepsUp.Count == 0)
        {
            AddModelNote("None of these can keep up with a live conversation on this PC. "
                         + "The fastest one is first.");
            foreach (var m in choices.OrderBy(m => m.LagMs <= 0 ? int.MaxValue : m.LagMs))
                ModelList.Items.Add(m);
            return;
        }

        // Everything keeps up, which is the ordinary case with a graphics card. A single
        // "recommended" heading over a list with no alternative is noise.
        if (lags.Count == 0)
        {
            foreach (var m in choices) ModelList.Items.Add(m);
            return;
        }

        AddModelNote("Recommended for this PC");
        foreach (var m in keepsUp) ModelList.Items.Add(m);
        AddModelNote("Slower on this PC. Captions arrive several seconds after the words are spoken.");
        foreach (var m in lags) ModelList.Items.Add(m);
    }

    /// <summary>
    /// A heading inside the list. Disabled so it is skipped by pointer and keyboard alike,
    /// which is the same trick the device picker's group headers use.
    /// </summary>
    private void AddModelNote(string text)
    {
        ModelList.Items.Add(new ListViewItem
        {
            Content = text,
            IsEnabled = false,
            FontSize = 12,
            FontWeight = Microsoft.UI.Text.FontWeights.SemiBold,
            Padding = new Thickness(4, 10, 4, 4),
            Margin = new Thickness(0),
            IsHitTestVisible = false,
        });
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

    private bool _loadingDevices;

    private async Task LoadDevicesAsync(bool fresh = false)
    {
        // One at a time. The constructor starts a fetch and the reconnect path starts another
        // when the picker is still empty, which on a normal cold start is simply because the
        // first is still polling — two loops then poll the same endpoint every 500 ms and both
        // populate. Harmless but wasteful, and it makes the trace hard to read.
        if (_loadingDevices) return;
        _loadingDevices = true;
        try
        {
            App.Trace($"LoadDevicesAsync start (fresh: {fresh})");
            // The backend needs a moment to bind its HTTP port on a cold start.
            for (var attempt = 0; attempt < 40; attempt++)
            {
                List<AudioDevice>? devices = null;
                try
                {
                    // fresh=1 makes the backend re-read the hardware in a child process
                    // rather than serving what PortAudio cached when it started. It costs
                    // about half a second, which is why it is only ever asked for by the
                    // refresh button and never on the startup path.
                    var url = "http://127.0.0.1:8765/devices.json" + (fresh ? "?fresh=1" : "");
                    var json = await _http.GetStringAsync(url);
                    devices = ParseDevices(json);
                    if (fresh && StaleFlagSet(json))
                    {
                        // The backend could not re-read and served its cached list instead.
                        // Not surfaced: a slightly out-of-date picker is not worth a warning
                        // bar, and the user can press the button again.
                        App.Trace("device refresh fell back to the cached list");
                    }
                }
                catch
                {
                    await Task.Delay(500);
                    continue;
                }

                // isRefresh carries one rule into PopulateDevices: a refresh may change what
                // the picker shows and what settings record, but must never change what is
                // being captured. Only the startup path is allowed to correct the selection,
                // because only there does correcting it mean anything but a restart.
                if (devices is { Count: > 0 })
                    _ui.TryEnqueue(() => PopulateDevices(devices, isRefresh: fresh));
                return;
            }
        }
        finally
        {
            // Cleared even on the give-up path, so a later reconnect can try again — that is
            // the whole point of the retry from OnConnection.
            _loadingDevices = false;
        }
    }

    private static bool StaleFlagSet(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            return doc.RootElement.TryGetProperty("stale", out var s)
                   && s.ValueKind == JsonValueKind.True;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Show that the input device is being changed, in the slot the refresh button occupies.
    ///
    /// Changing device restarts the backend and reloads the model, which is seconds of no
    /// captions. The centre panel cannot carry this: it is suppressed whenever a transcript is
    /// on screen, which is precisely when someone changes microphone mid-conversation, and
    /// showing it would cover the captions they already have. So the indicator lives beside the
    /// picker that started the wait.
    ///
    /// This matters more here than the same wait would elsewhere. A user relying on captions to
    /// follow a conversation cannot hear that the room is still talking; a transcript that has
    /// simply stopped is exactly what this app looks like when it has crashed. Every path that
    /// ends a restart has to come back through here — see OnBackendCrashed, which drops it
    /// before deciding anything, because two of its branches never reach the fatal banner.
    /// </summary>
    private void SetDeviceBusy(bool busy)
    {
        DeviceBusyRing.IsActive = busy;
        DeviceBusyRing.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
        // Hidden rather than disabled: the two share a slot, so a greyed button behind a
        // spinner would show through it.
        RefreshDevicesButton.Visibility = busy ? Visibility.Collapsed : Visibility.Visible;
    }

    /// <summary>
    /// Ask the backend to look at the hardware again.
    ///
    /// Windows hands the backend its device list once, when it starts, so a microphone
    /// plugged in afterwards does not exist as far as this app is concerned until something
    /// asks. That "something" is deliberately a button rather than a watcher: refreshing has
    /// to rebuild the picker, and a picker that rebuilds itself unprompted in the middle of a
    /// conversation is worse than one that waits to be asked.
    /// </summary>
    private async void OnRefreshDevices(object sender, RoutedEventArgs e)
    {
        if (_loadingDevices) return;

        RefreshDevicesButton.IsEnabled = false;
        try
        {
            await LoadDevicesAsync(fresh: true);
        }
        finally
        {
            RefreshDevicesButton.IsEnabled = true;
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
            // One flag, two names, because a microphone and an output endpoint are different
            // enough that the backend computes them separately — but the UI only ever asks
            // "is this the one Windows would have chosen", so they collapse here.
            var isDefault =
                (d.TryGetProperty("is_default_input", out var di) && di.ValueKind == JsonValueKind.True)
                || (d.TryGetProperty("is_default_output", out var dof) && dof.ValueKind == JsonValueKind.True);
            if (index >= 0 && !string.IsNullOrEmpty(name))
                result.Add(new AudioDevice(index, name!, api ?? string.Empty, loopback, isDefault));
        }
        return result;
    }

    private void PopulateDevices(List<AudioDevice> devices, bool isRefresh = false)
    {
        App.Trace($"PopulateDevices: {devices.Count} (refresh: {isRefresh})");
        _suppressDeviceEvent = true;
        try
        {
            DevicePicker.Items.Clear();

            // Defensive now, not load-bearing. The server narrows to the WASAPI enumeration on
            // Windows, so one entry per device normally arrives and there is nothing to collapse.
            // This still runs for the fallback list the server sends when WASAPI enumerates
            // nothing, where the same physical device does appear once per host API (MME,
            // DirectSound, WDM-KS) with the name mangled differently each time — MME truncates at
            // 31 characters, so one microphone arrives as "Microphone (Umik-1  Gain: 18dB",
            // "…18dB  )" and "…18dB)". Comparing letters and digits only, and treating a truncated
            // name as the same device, collapses them.
            //
            // It also carries an upgrading user across: a DeviceIndex saved when legacy entries
            // were still offered is matched back to its WASAPI twin by name, not by index.
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

        // Only now, with the event no longer suppressed, so that correcting the picker below
        // runs through the ordinary selection handler — on the startup path. A refresh passes
        // isRefresh so that correction stays suppressed, because there it would restart a
        // backend that is already captioning.
        ValidateRememberedDevice(isRefresh);
    }

    /// <summary>
    /// Check that the device index we launched on still means the device the user chose.
    ///
    /// PortAudio numbers devices by enumeration order, so the numbers move whenever the set of
    /// audio devices changes. Measured on one machine across two launches: the same Umik-1 went
    /// from index 30 to 27, and index 26 stopped meaning "Microphone (2- Logitech BRIO)" and
    /// started meaning "Headset (R-Phonak hearing aid)". Nothing errored. The app simply
    /// captioned a different microphone than the one on the table, which is the failure this
    /// product can least afford: it does not look broken, it just gets quietly worse.
    ///
    /// Validate rather than resolve. Capture has to start before the device list exists, because
    /// the list is served by the very process being started, so the index stays the fast path
    /// and this runs once the list arrives. When it matches, which is the overwhelmingly common
    /// case, nothing happens at all.
    ///
    /// <paramref name="isRefresh"/> carries the one rule a refresh must obey: it may change what
    /// the picker shows and what settings record, but never what is being captured. At startup
    /// correcting a rotted index means restarting a backend that has not begun captioning, which
    /// is right. On a refresh the backend is already holding an open stream on a real device —
    /// its capture cannot have moved just because an index did — so the same correction would
    /// buy nothing and cost a model reload, which is captions stopping mid-conversation.
    /// </summary>
    private void ValidateRememberedDevice(bool isRefresh = false)
    {
        var loopback = _settings.LoopbackDeviceIndex is not null;
        var wantedIndex = _settings.LoopbackDeviceIndex ?? _settings.DeviceIndex;
        var wantedName = loopback ? _settings.LoopbackDeviceName : _settings.DeviceName;

        // No remembered device: the system default is in use and there is nothing to check.
        if (wantedIndex is null) return;

        // Entries for the right kind of device only. /devices.json is a single flat array
        // holding two different index spaces — microphones are numbered by sounddevice and
        // loopback endpoints by pyaudiowpatch — and the ranges overlap, so index 27 exists in
        // both. Comparing across them would measure a microphone against a speaker.
        var candidates = DevicePicker.Items.OfType<ComboBoxItem>()
            .Select(i => new { Item = i, Entry = i.Tag as DeviceEntry })
            .Where(x => x.Entry is not null && x.Entry.Device.Loopback == loopback)
            .ToList();
        if (candidates.Count == 0) return;

        var atIndex = candidates.FirstOrDefault(x => x.Entry!.Aliases.Contains(wantedIndex.Value));

        // Upgrading from a build that only stored the index. An absent name is not evidence of
        // rot, it is evidence of an older settings file, so adopt whatever is there now and
        // start checking from the next launch. Warning here would fire on every existing
        // install, about a device that is working perfectly.
        if (string.IsNullOrEmpty(wantedName))
        {
            if (atIndex?.Entry is null) return;
            if (loopback) _settings.LoopbackDeviceName = atIndex.Entry.Device.Name;
            else _settings.DeviceName = atIndex.Entry.Device.Name;
            _settings.Save();
            // Index and outcome only. The device name is the string Diagnostics refuses to emit,
            // because "Headset (R-Phonak hearing aid)" discloses that someone wears a hearing
            // aid, and startup-trace.log survives on disk for the next launch.
            App.Trace($"device name adopted for existing setting at index {wantedIndex}");
            return;
        }

        var wantedKey = DeviceKey(CleanDeviceName(wantedName!));
        if (wantedKey.Length == 0) return;

        // Exact key comparison, deliberately not IsSameDevice.
        //
        // IsSameDevice tolerates a prefix match when the shorter name sits exactly on MME's
        // 31-character truncation boundary, and it decides that from the *raw* PortAudio name
        // length. The name stored here has already been cleaned and whitespace-collapsed, so its
        // length says nothing about truncation: passing it in would let any stored name that
        // happens to be 31 characters — "Microphone (HD Pro Webcam C920)" is exactly 31 — match
        // a longer device by prefix. That would silently select a different microphone, which is
        // the failure this whole change exists to remove.
        //
        // The keys being compared were both built by DeviceKey from cleaned names, so equality
        // is the right test and the truncation tolerance is neither needed nor safe.
        if (atIndex?.Entry is not null && atIndex.Entry.Key == wantedKey)
        {
            // The device is present and the index still means it. On a refresh that may be
            // news: the notice on screen could be a startup one saying this very device was
            // missing, and the button they just pressed is what fixed it. Leaving it up would
            // have the app insisting a device is unavailable while showing it selected — and
            // the "choose a microphone below" wording is unfollowable in that state, because
            // the device is already the selected item and re-selecting it raises no event.
            if (isRefresh) ClearDeviceNotice();
            return;   // index still means the right device, which is the usual case
        }

        var correct = candidates.FirstOrDefault(x => x.Entry!.Key == wantedKey);

        if (correct is null)
        {
            // The remembered device is not present at all: unplugged, powered off, or renamed by
            // a driver update. Do not silently substitute another one, which is the behaviour
            // being fixed here.
            App.Trace($"remembered device (index {wantedIndex}) not present in this enumeration");

            if (isRefresh)
            {
                // Both messages below describe what capture is doing, and both are written from
                // the startup path where the backend has just opened on the index it was given.
                // On a refresh the backend has been running for a while and this code cannot see
                // what it managed to open, so either sentence would be a guess. Telling a deaf
                // user captions are running when they are not is the single worst thing this
                // notice can do, so on this path it states only the part that is known.
                ShowDeviceNotice($"{wantedName} is not available. Choose a device below if you "
                                 + "want to switch.");
            }
            else if (atIndex?.Entry is not null)
            {
                // The index still resolves to a real device, so capture is running on that one.
                ShowDeviceNotice($"{wantedName} is not available, so Sunno is using "
                                 + $"{atIndex.Entry.Device.Name} instead.");
            }
            else
            {
                // The index resolves to nothing. Capture is not quietly falling back: the
                // backend tries every format against the stale index and then raises
                // MicrophoneOpenError, so nothing is being captured at all. Saying "Sunno is
                // listening to the default microphone" here would tell a deaf user captioning
                // was running when it was not, on the one path whose entire purpose is to stop
                // silent capture failures.
                ShowDeviceNotice($"{wantedName} is not available. Choose a microphone below to "
                                 + "start captioning.");
            }
            return;
        }

        App.Trace($"device index {wantedIndex} rotted; correcting to index {correct.Entry!.Device.Index}");

        if (isRefresh)
        {
            // Same correction, without the restart.
            //
            // The backend is already captioning from an open stream, which an index moving
            // underneath it cannot change — so there is nothing to restart it for, and doing so
            // would reload the model and stop captions mid-conversation for the crime of
            // plugging in an unrelated device.
            //
            // Both index/name pairs are updated, not just the microphone one. Someone captioning
            // system audio has their device recorded in LoopbackDeviceIndex, and leaving that to
            // rot while fixing only DeviceIndex would protect the path they are not using.
            //
            // Known gap, left deliberately: if the captured device is unplugged and a device with
            // an identical cleaned name appears in the same refresh, this rewrites the index to
            // the new one while the backend still holds the dead one, and says nothing because
            // the name matched. It needs two identically named devices and a swap between two
            // presses of the button. The honest fix is recovering capture when the active device
            // disappears, which is a larger change than a picker refresh.
            var newIndex = correct.Entry!.Device.Index;
            if (loopback)
            {
                _settings.LoopbackDeviceIndex = newIndex;
                _settings.LoopbackDeviceName = correct.Entry.Device.Name;
            }
            else
            {
                _settings.DeviceIndex = newIndex;
                _settings.DeviceName = correct.Entry.Device.Name;
            }
            // Without this the rewrite is decoration: the next launch would read the stale index
            // out of settings.json and hand it to the backend, which is the divergence this is
            // supposed to prevent, just deferred by one restart.
            _settings.Save();

            _suppressDeviceEvent = true;
            try
            {
                DevicePicker.SelectedItem = correct.Item;
            }
            finally
            {
                _suppressDeviceEvent = false;
            }

            // The device was found, so retire any notice claiming it was missing — including
            // the one this same method may have raised at startup, which is the case the
            // refresh button exists to resolve.
            ClearDeviceNotice();
            return;
        }

        // Hand this to the ordinary selection handler rather than restarting the backend here.
        // OnDeviceChanged does six things before it restarts — clears status, pauses the capture
        // clock, and resets _captureRequested, _connected, _engineReadyThisSession and
        // _startedPaused. Restarting directly at this point, which is seconds after launch and
        // after consent has already latched _captureRequested, would bring the backend up paused
        // with nothing left to un-pause it: no error on screen and no captions, ever.
        DevicePicker.SelectedItem = correct.Item;
    }

    /// <summary>
    /// Tell the user their remembered microphone is gone, without implying something broke.
    ///
    /// Informational rather than Warning: in the common case captions are still working, just
    /// from a different device than they picked. The point is that they find out, because the
    /// alternative is an app that quietly listens to the wrong microphone and seems to have got
    /// less accurate.
    ///
    /// Sticky, and that is the whole thing working. This fires while the engine is still
    /// loading, and OnStatus closes any non-sticky bar the moment "listening" arrives — which
    /// is precisely the frame on which the app starts captioning the wrong device. Without the
    /// flag the one visible output of the entire device-rot fix is dismissed by the app itself,
    /// a second before it matters.
    ///
    /// Declines to speak over a microphone problem. Those bars carry an actionable link
    /// ("Open Settings") and a fix; replacing that with a note about device names would leave
    /// the user with no route to the thing actually blocking them. Kept in _deviceNotice so the
    /// microphone path can put it back after it has finished with the bar.
    /// </summary>
    private void ShowDeviceNotice(string message)
    {
        _deviceNotice = message;
        // A dead engine outranks a changed microphone, and this has to be checked before the
        // two lines below rather than at the call sites. Nulling _crashDetail would throw away
        // the "Copy details" payload for the crash, and the notice itself is Informational —
        // painting "Microphone changed" over "the speech engine stopped" tells a user whose
        // captions have died that everything is fine. Reachable on a broken install: the
        // constructor raises the fatal banner, then LoadDevicesAsync finishes and device-rot
        // detection fires this.
        if (_backendFatal) return;
        _crashDetail = null;
        if (_micProblem) return;
        RenderDeviceNotice();
    }

    private void RenderDeviceNotice()
    {
        // Guarded here as well as in ShowDeviceNotice, so the rule lives with the thing it
        // protects rather than being re-stated at every caller. Same reason the Settings page
        // disables regions rather than enumerating what each one contains.
        if (_deviceNotice is null || _backendFatal) return;
        MicInfoBar.Severity = InfoBarSeverity.Informational;
        MicInfoBar.Title = "Microphone changed";
        MicInfoBar.Message = _deviceNotice;
        MicActionLink.Visibility = Visibility.Collapsed;
        MicInfoBar.IsOpen = true;
        _infoSticky = true;
    }

    /// <summary>
    /// The device notice, kept so it can be re-asserted.
    ///
    /// A microphone-permission problem outranks it and takes the bar over. Once the user fixes
    /// that, the recovery path closes the bar — and without this the device warning would be
    /// gone for good, having been raised milliseconds before the permission bar arrived.
    ///
    /// Retired by whatever resolves it: choosing a device (the remedy the notice asks for) or
    /// dismissing the bar. Without that it would outlive its own instruction — the user picks a
    /// working microphone, captions flow, and the bar still reads "not available, choose a
    /// microphone below", or worse still names a device that is no longer the one being
    /// captured.
    /// </summary>
    private string? _deviceNotice;

    /// <summary>Forget the device notice and take it off screen if it is the bar showing.</summary>
    private void ClearDeviceNotice()
    {
        if (_deviceNotice is null) return;
        _deviceNotice = null;
        if (MicInfoBar.IsOpen && !_micProblem && !_backendFatal)
        {
            _infoSticky = false;
            MicInfoBar.IsOpen = false;
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
        if (wanted is null)
        {
            SelectSystemDefaultDevice();
            return;
        }

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

    /// <summary>
    /// Name the microphone nobody chose.
    ///
    /// On a first run there is no saved device, so the picker used to sit on its placeholder
    /// while the backend quietly captured whatever Windows had set as the default. The app
    /// was working and unable to say so — for someone who cannot hear the room, "which
    /// microphone is this actually listening to" is not a curiosity, it is the difference
    /// between trusting a blank transcript and not.
    ///
    /// Selected under the existing event suppression, since this describes the device the
    /// backend already opened. Assigning it unsuppressed would hand it to OnDeviceChanged,
    /// which persists a choice the user never made and restarts a backend that is mid-launch.
    /// </summary>
    private void SelectSystemDefaultDevice()
    {
        foreach (var item in DevicePicker.Items.OfType<ComboBoxItem>())
        {
            if (item.Tag is not DeviceEntry entry) continue;
            // Microphones only. An output endpoint can be flagged as the default place sound
            // is played to, which is a different question from what to capture, and starting
            // a new user on their speakers would caption the room's silence.
            if (entry.Device.Loopback || !entry.Device.IsDefault) continue;
            DevicePicker.SelectedItem = item;
            ToolTipService.SetToolTip(DevicePicker,
                $"Captioning the microphone {entry.Device.Name}, your Windows default");
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
            // Marked in the label rather than only in the tooltip. This is the entry someone
            // lands on without choosing anything, and the one to come back to after trying
            // others, so it has to be findable with the list open and no pointer hovering.
            var item = new ComboBoxItem
            {
                Content = d.IsDefault ? $"{d.Name}  ·  Windows default" : d.Name,
                Tag = entry,
            };
            DevicePicker.Items.Add(item);
            ToolTipService.SetToolTip(item, d.Loopback
                ? $"Caption whatever is played through {d.Name}: calls, video, music"
                : $"{d.Name} ({d.HostApi})");
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

        // Choosing a device is the remedy the device notice asks for, so retire it here.
        // Otherwise it would survive its own instruction and keep naming a device that is no
        // longer the one being captured.
        ClearDeviceNotice();

        ToolTipService.SetToolTip(DevicePicker, device.Loopback
            ? $"Captioning system audio from {device.Name}"
            : $"Captioning the microphone {device.Name}");

        _settings.DeviceIndex = device.Loopback ? null : device.Index;
        _settings.LoopbackDeviceIndex = device.Loopback ? device.Index : null;
        // Record the name as well as the index. The index is what gets passed to the backend at
        // launch, but it is only valid for as long as the machine's device set is unchanged;
        // the name is what lets the next launch tell whether that index still means this device.
        // Taken from entry.Device.Name, which is the cleaned label the picker itself shows, so
        // it can be compared later with the same matching rules that built this list.
        _settings.DeviceName = device.Loopback ? null : device.Name;
        _settings.LoopbackDeviceName = device.Loopback ? device.Name : null;
        _settings.Save();

        // Switching capture device means restarting the backend; the model reload is the slow
        // part, so say so rather than appear hung.
        //
        // Restart, never Dispose+Start. Dispose tears down the job object permanently and
        // latches _stopping, so a Start afterwards leaves the new capture process untied to
        // kill-on-close (it would outlive a killed UI still holding the microphone) and with
        // crash reporting silently dead for the rest of the session.
        //
        // Two indicators, because neither covers both cases. The centre panel carries the
        // wait when there is no transcript yet, and is deliberately suppressed once there is
        // one, so it cannot cover captions the user already has. The ring by the picker
        // covers the case the centre panel will not: a device changed in the middle of a
        // conversation, where the only other feedback is the transcript stopping.
        ClearStatus();
        PauseCaptureClock();
        ShowLoadingState($"Switching to {device.Name}");
        SetDeviceBusy(true);

        _captureRequested = false;
        _connected = false;
        _engineReadyThisSession = false;
        _startedPaused = !_micGranted || _micDeclined;
        ClearBackendFatal();   // see SwitchModelAsync: a deliberate restart un-kills the engine

        var error = _backend.Restart(
            device: device.Loopback ? null : device.Index.ToString(),
            model: _settings.Model,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused,
            loopbackDevice: device.Loopback ? device.Index : null,
            computeDevice: _settings.ForceCpu ? "cpu" : "auto");
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

    /// <summary>
    /// Window-wide keyboard shortcuts.
    ///
    /// Registered on the content root rather than on the menu items they belong to: an
    /// accelerator declared on a MenuFlyoutItem only fires while that flyout is open, because
    /// the flyout's content is not in the visual tree until it is shown. The menu items carry
    /// KeyboardAcceleratorTextOverride so the shortcut is still discoverable where a user
    /// would look for it.
    ///
    /// Caption size is the shortcut that matters here. It is adjusted mid-conversation by
    /// someone using the app to follow that conversation, and reaching for a menu costs them
    /// the sentence being spoken while they do it.
    /// </summary>
    private void RegisterShortcuts(FrameworkElement root)
    {
        // WinUI shows an accelerator's key combination as a tooltip on whichever element owns
        // it, and the default placement mode is Auto, meaning "show it". These are owned by the
        // content root because they must work window-wide - so the tooltip appeared on hover
        // anywhere in the app, reading "Ctrl++" because Ctrl + the OEM plus key formats that
        // way. It also outlived the thing that spawned it and sat over the transcript.
        //
        // This is the actual source. Two earlier attempts blamed the menu's
        // KeyboardAcceleratorTextOverride and neither removed it, because the tooltip was never
        // coming from the menu at all.
        root.KeyboardAcceleratorPlacementMode = KeyboardAcceleratorPlacementMode.Hidden;

        // The +/- on the main keyboard row are OEM keys, not VirtualKey.Add/Subtract, which
        // are the numeric keypad. Both are registered, because "Ctrl and the plus key" means
        // whichever one the user's hand is nearest - and on most layouts the main-row one
        // needs Shift to produce an actual "+", so the unshifted key is what to listen for.
        const Windows.System.VirtualKey OemPlus = (Windows.System.VirtualKey)187;
        const Windows.System.VirtualKey OemMinus = (Windows.System.VirtualKey)189;

        Add(OemPlus, () => SetFontSize(CaptionSize + 3));
        Add(Windows.System.VirtualKey.Add, () => SetFontSize(CaptionSize + 3));
        Add(OemMinus, () => SetFontSize(CaptionSize - 3));
        Add(Windows.System.VirtualKey.Subtract, () => SetFontSize(CaptionSize - 3));

        // Escape leaves Settings, from wherever focus is inside the page.
        var escape = new KeyboardAccelerator { Key = Windows.System.VirtualKey.Escape };
        escape.Invoked += (_, args) =>
        {
            if (SettingsPage.Visibility != Visibility.Visible) return;
            CloseSettings();
            args.Handled = true;
        };
        root.KeyboardAccelerators.Add(escape);

        void Add(Windows.System.VirtualKey key, Action action)
        {
            var accelerator = new KeyboardAccelerator
            {
                Key = key,
                Modifiers = Windows.System.VirtualKeyModifiers.Control,
            };
            accelerator.Invoked += (_, args) =>
            {
                // Marked handled either way: while an overlay is up the window behind it is
                // inert, and letting the keystroke fall through would resize a transcript the
                // user cannot see.
                args.Handled = true;
                if (SettingsPage.Visibility == Visibility.Visible) return;
                if (SetupOverlay.Visibility == Visibility.Visible) return;
                action();
            };
            root.KeyboardAccelerators.Add(accelerator);
        }
    }

    private void SetFontSize(double size)
    {
        CaptionSize = Math.Clamp(size, MinFont, MaxFont);
        // The transcript ItemsControl binds its FontSize to CaptionSize (OneWay) and the caption
        // RichTextBlocks inherit it, so on-screen lines resize in place — no teardown, no lost
        // selection. Persist the choice so it survives a restart.
        _settings.CaptionFontSize = CaptionSize;
        _settings.Save();
        ScrollToEnd();
    }

    private void OnToggleAlwaysOnTop(object sender, RoutedEventArgs e)
    {
        if (AppWindow.Presenter is OverlappedPresenter presenter)
            presenter.IsAlwaysOnTop = AlwaysOnTopItem.IsChecked;

        // Persisted, which it previously was not. AppSettings had an AlwaysOnTop property that
        // nothing ever wrote: the window hardcoded it true on load and the menu item hardcoded
        // IsChecked="True" in XAML, so turning it off lasted exactly until the next launch and
        // the diagnostics report said "True" whatever the user had chosen.
        _settings.AlwaysOnTop = AlwaysOnTopItem.IsChecked;
        _settings.Save();
    }

    private void OnClear(object sender, RoutedEventArgs e)
    {
        Lines.Clear();
        _provisional = null;
        _currentUtterance = -1;
        SetEmptyStateVisible(true);

        // The transcript is the conversation, so clearing it starts a new one and the timer
        // begins again. This is the only place the count is thrown away — pausing, switching
        // device and reloading the model all keep it.
        var wasRunning = _captureClock.IsRunning;
        ResetCaptureClock();
        if (wasRunning) ShowElapsed();
    }

    private void OnSpeakerClicked(object sender, ItemClickEventArgs e)
    {
        if (e.ClickedItem is SpeakerRow row) _ = ShowSpeakerDialogAsync(row);
    }

    /// <summary>
    /// The row a context menu item was invoked on. The flyout lives inside the item template,
    /// so its DataContext is that row - read it from the menu item rather than from the list's
    /// selection, which right-clicking does not necessarily change.
    /// </summary>
    private static SpeakerRow? RowFromMenu(object sender) =>
        (sender as FrameworkElement)?.DataContext as SpeakerRow;

    private void OnSpeakerEditRequested(object sender, RoutedEventArgs e)
    {
        if (RowFromMenu(sender) is SpeakerRow row) _ = ShowSpeakerDialogAsync(row);
    }

    private void OnSpeakerDeleteRequested(object sender, RoutedEventArgs e)
    {
        if (RowFromMenu(sender) is SpeakerRow row) _ = DeleteSpeakerAsync(row);
    }

    /// <summary>
    /// Name a speaker, mark them as the user, or merge two speakers. Merge exists because
    /// automatic labelling sometimes splits one person across two labels.
    /// </summary>
    private async Task ShowSpeakerDialogAsync(SpeakerRow row)
    {
        var originalName = row.Named ? row.Label.Replace(" (You)", string.Empty) : string.Empty;
        var nameBox = new TextBox
        {
            Header = "Name",
            PlaceholderText = "e.g. Priya",
            Text = originalName,
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

        // Names the outcome. "Same person as: Marco" reads as though Marco is being changed,
        // when in fact this speaker is the one that disappears - so say which name survives
        // before the user commits to it.
        var who = string.IsNullOrEmpty(originalName) ? "This speaker" : originalName;
        var mergeHint = new TextBlock
        {
            Text = $"{who} will be removed from the list, and their lines will move to "
                   + "whoever you choose here.",
            TextWrapping = TextWrapping.Wrap,
            Opacity = 0.7,
            Visibility = Visibility.Collapsed,
        };
        // Only once a target is actually picked: with "Nobody" selected, nothing is being
        // removed and the warning would describe something that is not going to happen.
        merge.SelectionChanged += (_, _) =>
            mergeHint.Visibility = merge.SelectedIndex > 0
                ? Visibility.Visible
                : Visibility.Collapsed;

        var panel = new StackPanel { Spacing = 12, Width = 320 };
        panel.Children.Add(nameBox);
        panel.Children.Add(isSelf);
        panel.Children.Add(hint);
        panel.Children.Add(merge);
        panel.Children.Add(mergeHint);

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

        // Sent whenever the box differs from what was in it, so emptying it removes the name
        // rather than being read as "no change". The backend already treats an empty string
        // as unname-and-unpin; this guard was the only thing stopping a user from undoing a
        // name they had typed by mistake.
        var name = nameBox.Text.Trim();
        if (name != originalName) await _client.RenameSpeakerAsync(row.Id, name);
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
