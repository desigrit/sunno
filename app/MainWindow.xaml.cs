using System;
using System.Collections.ObjectModel;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using LiveCaptions.Models;
using LiveCaptions.Services;
using Microsoft.UI.Composition.SystemBackdrops;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Windows.Graphics;
using Windows.Security.Authorization.AppCapabilityAccess;

namespace LiveCaptions;

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
    /// <summary>Whether the InfoBar's action can still raise the dialog, or must fall back to
    /// Settings because Windows will not prompt a second time.</summary>
    private bool _micCanPrompt;

    /// <summary>Caption text size; the item templates read this.</summary>
    public static double CaptionSize { get; private set; } = 26;

    public ObservableCollection<CaptionLine> Lines { get; } = new();
    public ObservableCollection<SpeakerRow> Speakers { get; } = new();

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

        Closed += (_, _) =>
        {
            MicrophoneAccess.Changed -= OnMicAccessChanged;
            _ = _client.DisposeAsync();
            _backend.Dispose();
            // WinUI doesn't end the process when the last window closes; without this the app
            // lingers invisibly (and, before the job object, kept the microphone open).
            Application.Current.Exit();
        };

        // Consent has to be settled before anything opens the microphone. When access isn't
        // already granted the backend starts paused, so the ~33 s model load overlaps the
        // consent dialog rather than following it, and the backend's own mic_denied banner
        // can't race the system prompt with a second, contradictory explanation.
        var micStatus = MicrophoneAccess.Check();
        _micGranted = micStatus is null or AppCapabilityAccessStatus.Allowed;
        _startedPaused = !_micGranted;

        var error = _backend.Start(
            device: _settings.DeviceIndex?.ToString(),
            model: _settings.Model,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused);
        if (!string.IsNullOrEmpty(error)) StatusText.Text = error;
        _client.Start();
        _ = LoadDevicesAsync();

        // The dialog is system-owned UI and RequestAccessAsync must run on the UI thread, so it
        // waits for real activation instead of firing from the constructor, where there is no
        // visible window for it to sit in front of.
        Activated += OnFirstActivated;
    }

    /// <summary>Mica, extended title bar and a medium default size, matching inbox apps.</summary>
    private void ConfigureWindow()
    {
        Title = "Live Captions";

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
            MicInfoBar.IsOpen = false;
        }

        if (_micProblem && st.State == "stopped")
        {
            // Keep the real reason on screen rather than the generic paused text.
            return;
        }

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
                    "Windows is blocking microphone access for Live Captions, so nothing can " +
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

    private async void OnMicAction(object sender, RoutedEventArgs e)
    {
        // The same button means different things depending on whether Windows will still
        // prompt: asking again is useless once the answer has been recorded.
        if (_micCanPrompt)
        {
            ApplyMicrophoneStatus(await MicrophoneAccess.RequestAsync());
            return;
        }
        await Windows.System.Launcher.LaunchUriAsync(
            new Uri("ms-settings:privacy-microphone"));
    }

    private void OnFirstActivated(object sender, WindowActivatedEventArgs e)
    {
        if (_micPromptDone || e.WindowActivationState == WindowActivationState.Deactivated) return;
        _micPromptDone = true;
        Activated -= OnFirstActivated;
        _ = EnsureMicrophoneAccessAsync();
    }

    /// <summary>
    /// Settle microphone consent, prompting when Windows has never asked.
    ///
    /// This is the payoff of shipping as MSIX. The capture itself runs in the Python child over
    /// WASAPI, which the privacy gate silently denies rather than prompting on, so the dialog
    /// has to be raised deliberately from the packaged UI process.
    /// </summary>
    private async Task EnsureMicrophoneAccessAsync()
    {
        var status = MicrophoneAccess.Check();

        // UserPromptRequired means "never asked", not "refused" — the one state a dialog exists
        // for, and the one the previous CheckAccess-only code mislabelled as a denial.
        if (status == AppCapabilityAccessStatus.UserPromptRequired)
            status = await MicrophoneAccess.RequestAsync();

        ApplyMicrophoneStatus(status);

        // Notice a later grant from Settings, so the fallback isn't a dead end that needs a relaunch.
        MicrophoneAccess.Changed += OnMicAccessChanged;
    }

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
                    "Live Captions needs your microphone to transcribe what people say. " +
                    "Audio is processed on this PC and never leaves it.";
                break;

            case AppCapabilityAccessStatus.DeniedByUser:
                _micCanPrompt = false;
                MicInfoBar.Title = "Microphone access is off";
                MicInfoBar.Message =
                    "Microphone access for Live Captions is turned off, so nothing can be " +
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
        if (!_micGranted || !_startedPaused || _captureRequested || !_connected) return;
        _captureRequested = true;
        _ = _client.StartCaptureAsync();
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
            return;
        }
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
        DownloadBar.IsIndeterminate = false;
        DownloadBar.Value = Math.Clamp(p.Percent, 0, 100);
        DownloadText.Text =
            $"{p.Downloaded / 1048576.0:0} MB of {p.Total / 1048576.0:0} MB · {p.Percent:0}%";
    }

    private void OnDownloadComplete()
    {
        DownloadBar.Value = 100;
        DownloadText.Text = "Done. Loading the speech engine…";
        SetupOverlay.Visibility = Visibility.Collapsed;
        ModelList.IsEnabled = true;
    }

    private void OnDownloadFailed(string message)
    {
        SetupError.Title = "Download failed";
        SetupError.Message = message;
        SetupError.IsOpen = true;
        DownloadPanel.Visibility = Visibility.Collapsed;
        DownloadButton.IsEnabled = true;
        ModelList.IsEnabled = true;
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

    private async void OnToggleCapture(object sender, RoutedEventArgs e) =>
        await _client.ToggleAsync();

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
