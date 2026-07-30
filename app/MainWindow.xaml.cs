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
using Microsoft.UI.Xaml.Media;
using Windows.Graphics;
using Windows.Security.Authorization.AppCapabilityAccess;

namespace Sunno;

/// <summary>A microphone the backend can capture from.</summary>
public sealed record AudioDevice(int Index, string Name, string HostApi);

/// <summary>A model shown in first-run setup.</summary>
public sealed record ModelChoice(string Id, string Name, string Detail, int ApproxMb, bool Available)
{
    public string SizeLabel => ApproxMb >= 1024
        ? $"{ApproxMb / 1024.0:0.0} GB"
        : $"{ApproxMb} MB";
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
    /// <summary>The model to fall back to when a stored choice turns out not to load.</summary>
    private const string DefaultModel = "large-v3";
    /// <summary>Keeps an explanatory notice up until the user dismisses it themselves.</summary>
    private bool _infoSticky;
    /// <summary>Whether the InfoBar's action can still raise the dialog, or must fall back to
    /// Settings because Windows will not prompt a second time.</summary>
    private bool _micCanPrompt;
    /// <summary>The backend died; stop reporting progress that will never happen.</summary>
    private bool _backendFatal;
    /// <summary>Set while the engine is reloading onto a different model.</summary>
    private string? _switchingTo;
    /// <summary>Suppresses the Checked handler while the list is rebuilt from the backend.</summary>
    private bool _suppressModelEvent;

    /// <summary>Caption text size; the item templates read this.</summary>
    public static double CaptionSize { get; private set; } = 26;

    public ObservableCollection<CaptionLine> Lines { get; } = new();
    public ObservableCollection<SpeakerRow> Speakers { get; } = new();
    public ObservableCollection<ModelRow> Models { get; } = new();

    public MainWindow()
    {
        InitializeComponent();
        _ui = DispatcherQueue.GetForCurrentThread();

        ConfigureWindow();

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
            MicrophoneAccess.Changed -= OnMicAccessChanged;
            _ = _client.DisposeAsync();
            _backend.Dispose();
            // WinUI doesn't end the process when the last window closes; without this the app
            // lingers invisibly (and, before the job object, kept the microphone open).
            Application.Current.Exit();
        };

        // Consent has to be settled before anything opens the microphone. On first run we haven't
        // asked yet, so the backend starts paused regardless of what Windows reports — otherwise
        // the microphone would already be live behind our own consent dialog, which would make
        // asking dishonest.
        var micStatus = MicrophoneAccess.Check();
        _lastGoodModel = _settings.Model;
        _micGranted = (micStatus is null or AppCapabilityAccessStatus.Allowed)
                      && _settings.MicrophoneAsked;
        _startedPaused = !_micGranted;

        var error = _backend.Start(
            device: _settings.DeviceIndex?.ToString(),
            model: _settings.Model,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused);
        if (!string.IsNullOrEmpty(error)) StatusText.Text = error;
        _client.Start();
        _ = LoadDevicesAsync();

        // Consent is asked from the content's Loaded event rather than window activation: a
        // window that is shown without being focused still needs to ask, otherwise the backend
        // sits paused behind a dialog that never appears.
        if (Content is FrameworkElement root)
        {
            root.Loaded += (_, _) =>
            {
                if (_micPromptDone) return;
                _micPromptDone = true;
                _ = EnsureMicrophoneAccessAsync();
            };
        }
    }

    /// <summary>Mica, extended title bar and a medium default size, matching inbox apps.</summary>
    private void ConfigureWindow()
    {
        Title = "Sunno";

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
        line.IsFinal = isFinal;
    }

    private void OnDiscarded(int id)
    {
        if (_provisional is not null && _currentUtterance == id) Lines.Remove(_provisional);
        _provisional = null;
        _currentUtterance = -1;
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
        if (!_running) return;
        LevelBar.Value = Math.Clamp((lv.Db + 60) / 60.0, 0, 1) * 100;
    }

    private void OnStatus(StatusEvent st)
    {
        if (st.Running is bool running) SetRunning(running);
        _backendLoading = st.State == "loading";

        if (st.State == "error")
        {
            ShowActionableError(st);
            StatusText.Text = st.Code == "mic_denied" ? "Microphone blocked" : "Error";
            return;
        }

        // Only a successful capture clears the banner. The backend pauses itself after a
        // microphone failure and immediately reports "stopped", so clearing on every status
        // would erase the explanation milliseconds after showing it — leaving a message that
        // reads as if the user had stopped capture themselves.
        if (st.State == "listening")
        {
            _micProblem = false;
            // A sticky notice explains something the user did that didn't take effect, so it
            // outlives the recovery it describes — otherwise the model silently snaps back
            // with no explanation at all.
            if (!_infoSticky) MicInfoBar.IsOpen = false;
            ShowReadyState();
        }

        if (_micProblem && st.State == "stopped")
        {
            // Keep the real reason on screen rather than the generic paused text.
            return;
        }

        // A spinner beside "Stopped" reads as "still working on it". Once the backend reports
        // it is simply paused, the engine is up and the wait is over.
        if (st.State == "stopped") ShowIdleState();

        StatusText.Text = st.State switch
        {
            "loading" => $"Loading {st.Model}…",
            "stopped" => "Stopped · microphone released",
            "listening" => ShortDeviceName(st.Device) ?? "Listening",
            _ => st.State,
        };
    }

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
        _backendLoading = false;

        // A crash while switching means the new model never came up. Fall back to the last
        // model known to load instead of leaving the app dead — and never persist the choice
        // that broke it, or every future launch would reload it and crash again.
        if (_switchingTo is { } failed)
        {
            _switchingTo = null;
            _awaitingSwitchReconnect = false;
            foreach (var m in Models) { m.IsBusy = false; m.Refresh(); }
            SelectModelRow(_lastGoodModel);

            if (!_recoveringModel && failed != _lastGoodModel && _lastGoodModel.Length > 0)
            {
                _recoveringModel = true;
                var failedName = Models.FirstOrDefault(m => m.Id == failed)?.Name ?? failed;

                MicInfoBar.Severity = InfoBarSeverity.Warning;
                MicInfoBar.Title = $"{failedName} couldn't be loaded";
                MicInfoBar.Message = "Switching back to the model that was working.";
                MicActionLink.Visibility = Visibility.Collapsed;
                MicInfoBar.IsOpen = true;
                _infoSticky = true;

                _ = SwitchModelAsync(_lastGoodModel);
                return;
            }
        }
        else if (!_recoveringModel && _settings.Model != DefaultModel && Models.Count > 0)
        {
            // Crashing outside a switch, on a model we persisted, means the stored choice
            // itself is bad — the state an interrupted or unverified switch can leave behind.
            // Fall back to the recommended model once instead of crashing identically on
            // every future launch, which this user has no way to escape from inside the app.
            _recoveringModel = true;
            _settings.Model = DefaultModel;
            _settings.Save();
            _lastGoodModel = DefaultModel;

            MicInfoBar.Severity = InfoBarSeverity.Warning;
            MicInfoBar.Title = "That model wouldn't load";
            MicInfoBar.Message = "Falling back to Whisper large-v3.";
            MicActionLink.Visibility = Visibility.Collapsed;
            MicInfoBar.IsOpen = true;
            _infoSticky = true;

            _ = SwitchModelAsync(DefaultModel);
            return;
        }

        _backendFatal = true;
        StatusText.Text = "Speech engine stopped";

        _micProblem = false;
        _micCanPrompt = false;
        MicInfoBar.Severity = InfoBarSeverity.Error;
        MicInfoBar.Title = "The speech engine stopped";
        MicInfoBar.Message = $"{message}\n\nDetails were written to {BackendHost.LogPath}";
        MicActionLink.Content = "Copy details";
        MicActionLink.Visibility = Visibility.Visible;
        MicInfoBar.IsOpen = true;
        _crashDetail = $"{message}\n\nLog: {BackendHost.LogPath}";
    }

    private string? _crashDetail;

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
            // Recovering from our own "not now" needs no OS round trip — the OS never denied us.
            if (_micDeclined)
            {
                _micDeclined = false;
                ApplyMicrophoneStatus(MicrophoneAccess.Check());
                return;
            }
            ApplyMicrophoneStatus(await MicrophoneAccess.RequestAsync());
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
        var status = MicrophoneAccess.Check();

        // UserPromptRequired means "never asked", not "refused" — the state the previous
        // CheckAccess-only code mislabelled as a denial.
        if (status == AppCapabilityAccessStatus.UserPromptRequired)
            status = await MicrophoneAccess.RequestAsync();

        if (!_settings.MicrophoneAsked && status is null or AppCapabilityAccessStatus.Allowed)
            await AskForMicrophoneAsync();

        ApplyMicrophoneStatus(status);

        // Notice a later grant from Settings, so the fallback isn't a dead end that needs a relaunch.
        MicrophoneAccess.Changed += OnMicAccessChanged;
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
        _ui.TryEnqueue(() => ApplyMicrophoneStatus(MicrophoneAccess.Check()));

    /// <summary>
    /// Render a consent status. Each denial has a different remedy, so they can't share one
    /// message: pointing at the per-app toggle is actively misleading when the device-wide one
    /// is off, because the per-app control isn't even shown in that state.
    /// </summary>
    private void ApplyMicrophoneStatus(AppCapabilityAccessStatus? status)
    {
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
    private void OnMicInfoClosed(InfoBar sender, object args) => _infoSticky = false;

    private void OnModelSectionExpanding(Expander sender, ExpanderExpandingEventArgs args) =>
        HeaderModelName.Visibility = Visibility.Collapsed;

    private void OnModelSectionCollapsed(Expander sender, ExpanderCollapsedEventArgs args) =>
        HeaderModelName.Visibility = Visibility.Visible;

    /// <summary>Keep the collapsed header's summary in step with what's actually loaded.</summary>
    private void UpdateHeaderModelName()
    {
        var active = Models.FirstOrDefault(m => m.IsSelected);
        HeaderModelName.Text = active?.Name ?? string.Empty;
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
                    IsSelected = o.Id == current,
                };
                row.Refresh();
                if (row.IsSelected) row.Status = "In use";
                Models.Add(row);
            }
        }
        finally
        {
            _suppressModelEvent = false;
        }
        UpdateHeaderModelName();
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
        _startedPaused = !_micGranted || _micDeclined;
        _awaitingSwitchReconnect = true;

        // Deliberately NOT persisted yet. A model that downloads but fails to load would
        // otherwise become the choice reloaded on every future launch, turning one bad switch
        // into a crash loop with no way out from inside the app.
        var error = _backend.Restart(
            device: _settings.DeviceIndex?.ToString(),
            model: id,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused);

        if (!string.IsNullOrEmpty(error))
        {
            _switchingTo = null;
            _awaitingSwitchReconnect = false;
            if (row is not null) { row.IsBusy = false; row.Refresh(); }
            SelectModelRow(_lastGoodModel);
            StatusText.Text = error;
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

    /// <summary>Device strings carry format detail that's too long for the status line.</summary>
    private static string? ShortDeviceName(string? device)
    {
        if (string.IsNullOrEmpty(device)) return null;
        var cut = device.IndexOf('(');
        return cut > 1 ? device[..cut].Trim() : device;
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

            // A reconnect after a restart is the switch completing. Guarded on the restart flag
            // rather than on _switchingTo alone: during a long download the socket is still on
            // the old backend, and a transient drop there would otherwise be reported as a
            // successful switch that never happened.
            if (_awaitingSwitchReconnect && _switchingTo is { } finished)
            {
                _awaitingSwitchReconnect = false;
                _switchingTo = null;
                _recoveringModel = false;

                // Only now is the choice known to work, so only now is it safe to persist.
                _lastGoodModel = finished;
                _settings.Model = finished;
                _settings.Save();

                var row = Models.FirstOrDefault(m => m.Id == finished);
                if (row is not null)
                {
                    row.Available = true;
                    row.Refresh();
                    row.Status = "In use";
                }
                SelectModelRow(finished);
                foreach (var other in Models.Where(m => m.Id != finished)) other.Refresh();
            }
            else if (Models.Count == 0)
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
        StatusText.Text = _backendLoading ? "Starting the speech engine…" : "Reconnecting…";
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
            LevelBar.Value = 0;
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
            ModelList.Items.Add(new ModelChoice(o.Id, o.Name, o.Detail, o.ApproxMb, o.Available));

        // Preselect the first already-downloaded model, otherwise the recommended one.
        var preferred = ModelList.Items
            .OfType<ModelChoice>()
            .FirstOrDefault(m => m.Available);
        ModelList.SelectedItem = preferred ?? ModelList.Items.FirstOrDefault();

        SetupError.IsOpen = false;
        DownloadPanel.Visibility = Visibility.Collapsed;
        DownloadButton.IsEnabled = true;
        SetupOverlay.Visibility = Visibility.Visible;
        StatusText.Text = "Setup required";
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
            foreach (var m in Models) m.IsSelected = m.Id == id;
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
            if (index >= 0 && !string.IsNullOrEmpty(name))
                result.Add(new AudioDevice(index, name!, api ?? string.Empty));
        }
        return result;
    }

    private void PopulateDevices(List<AudioDevice> devices)
    {
        _suppressDeviceEvent = true;
        try
        {
            DevicePicker.Items.Clear();
            foreach (var d in devices)
            {
                DevicePicker.Items.Add(new ComboBoxItem
                {
                    Content = d.Name,
                    Tag = d.Index,
                    // The same physical mic shows up under several host APIs; the tooltip
                    // tells them apart without cluttering the closed state.
                    });
                ToolTipService.SetToolTip(
                    (ComboBoxItem)DevicePicker.Items[^1], $"{d.Name} — {d.HostApi}");
            }
        }
        finally
        {
            _suppressDeviceEvent = false;
        }
    }

    private void OnDeviceChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressDeviceEvent) return;
        if (DevicePicker.SelectedItem is not ComboBoxItem { Tag: int index }) return;

        _settings.DeviceIndex = index;
        _settings.Save();

        // Switching capture device means restarting the backend; the model reload is the slow
        // part, so say so rather than appear hung.
        StatusText.Text = "Switching microphone…";
        _backend.Dispose();
        var error = _backend.Start(device: index.ToString(), model: _settings.Model,
                                   vocabulary: _settings.Vocabulary);
        if (!string.IsNullOrEmpty(error)) StatusText.Text = error;
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
