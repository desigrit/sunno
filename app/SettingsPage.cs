using System.Collections.Generic;
using System.IO;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Sunno.Services;

namespace Sunno;

/// <summary>
/// The Settings page: what build am I running, how do I send that to whoever wrote this, and
/// the handful of choices that are not made mid-conversation.
///
/// A full-window page rather than a dialog, matching the inbox Windows apps. It is an overlay
/// in MainWindow.xaml rather than a Frame: the app is a single window with no navigation stack,
/// and SetupOverlay already establishes the pattern.
///
/// Caption size and always-on-top deliberately stay in the overflow menu. They are one-click
/// adjustments made in the middle of a conversation by someone using this app to follow that
/// conversation; moving them behind a page would be a regression for the person it exists for.
/// </summary>
public sealed partial class MainWindow
{
    /// <summary>What the diagnostics box is currently showing, which is exactly what Copy
    /// copies and Save writes. Held so the three can never diverge.</summary>
    private string _diagnosticsText = string.Empty;

    private void OnOpenSettings(object sender, RoutedEventArgs e)
    {
        // Setup wins. MoreButton is only painted over during first run, not disabled, so
        // without this the page could open on top of setup and closing it would drop the
        // user back into a half-finished download.
        //
        // Not when it is only a placeholder for backend startup, though. That screen asks
        // nothing and there is nothing to interrupt, and it covers the whole window, so
        // treating it like a real prompt would take away the one route to the diagnostics
        // export at exactly the moment a backend that will not start makes it worth having.
        if (SetupOverlay.Visibility == Visibility.Visible && !_setupProvisional) return;

        RefreshDiagnostics();

        ClarityToggle.IsOn = _settings.ShowClarity;
        ForceCpuToggle.IsOn = _settings.ForceCpu;
        RefreshRecordingsPath();
        AboutVersion.Text = $"Sunno {Diagnostics.AppVersion()}";
        NoSpeakersInSettings.Visibility = Speakers.Count == 0
            ? Visibility.Visible
            : Visibility.Collapsed;

        SetSettingsOpen(true);
    }

    private void OnCloseSettings(object sender, RoutedEventArgs e) => CloseSettings();

    private void CloseSettings()
    {
        if (SettingsPage.Visibility != Visibility.Visible) return;
        SetSettingsOpen(false);

        // Captions kept arriving behind the page and the user could not see them, so land
        // them back at the newest line rather than wherever they were reading before.
        ScrollToEnd();
        MoreButton.Focus(FocusState.Programmatic);
    }

    /// <summary>
    /// Show or hide the page, and take the rest of the window out of the tab order while it is
    /// up.
    ///
    /// A visible overlay does not stop Tab in WinUI: without this the device picker, the
    /// pause button, the transcript and the error banner all stay reachable behind the page.
    /// Someone driving the app by keyboard or a screen reader would be operating controls they
    /// cannot see.
    ///
    /// Listed control by control rather than by container: IsEnabled belongs to Control, and
    /// the two containers here are a Grid and a Border, neither of which is one. IsHitTestVisible
    /// covers those for the pointer, but only disabling the controls themselves takes them out
    /// of the tab order.
    /// </summary>
    private void SetSettingsOpen(bool open)
    {
        SettingsPage.Visibility = open ? Visibility.Visible : Visibility.Collapsed;

        BodyArea.IsHitTestVisible = !open;
        CommandArea.IsHitTestVisible = !open;

        foreach (var control in new Control[]
                 {
                     SpeakerList, ModelToggle, ModelPanel, CaptionScroller,
                     DevicePicker, ToggleButton, MoreButton, MicInfoBar,
                     CompactEnterButton, RefreshDevicesButton,
                 })
        {
            control.IsEnabled = !open;
        }

        if (open) SettingsBackButton.Focus(FocusState.Programmatic);
    }

    /// <summary>
    /// Rebuild the report. Called every time the page opens, never cached: a page opened after
    /// the engine has died must not show the state it had when the app started.
    /// </summary>
    private void RefreshDiagnostics()
    {
        try
        {
            _diagnosticsText = Diagnostics.BuildExport(_settings, _activeModel, _computeDevice,
                                                       _backend.IsRunning, _connected, _crashDetail);
        }
        catch (Exception ex)
        {
            _diagnosticsText = $"Could not build the report: {ex.GetType().Name}: {ex.Message}";
        }

        // AcceptsReturn is already true from the markup. Assigning to a TextBox that has it
        // false keeps only the text up to the first line break, which would show one line in
        // the control whose whole job is letting the user read what they are about to share.
        DiagnosticsBox.Text = _diagnosticsText;
        DiagnosticsStatus.Visibility = Visibility.Collapsed;
    }

    private void SetDiagnosticsStatus(string message)
    {
        DiagnosticsStatus.Text = message;
        DiagnosticsStatus.Visibility = Visibility.Visible;
    }

    /// <summary>
    /// Copy, and say so on screen either way.
    ///
    /// The clipboard genuinely fails here: a COMException from SetContent is sitting in this
    /// machine's startup-error.log. No retry loop — this thread is STA and the clipboard needs
    /// its message pump running for the current owner to let go, so sleeping on it would make a
    /// second attempt less likely to succeed than the first. The box is selectable, so Ctrl+C
    /// is the fallback and the message says so.
    /// </summary>
    private void OnCopyDiagnostics(object sender, RoutedEventArgs e)
    {
        try
        {
            var package = new Windows.ApplicationModel.DataTransfer.DataPackage();
            package.SetText(_diagnosticsText);
            Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(package);
        }
        catch (Exception ex)
        {
            App.Trace($"clipboard copy failed: {ex.GetType().Name}");
            SetDiagnosticsStatus("Windows would not let Sunno use the clipboard. "
                                 + "Select the text above and press Ctrl+C instead.");
            return;
        }

        try
        {
            // Without this the text is gone from the clipboard as soon as Sunno closes, which
            // is exactly what someone does after copying a report to paste into a bug report.
            Windows.ApplicationModel.DataTransfer.Clipboard.Flush();
        }
        catch (Exception ex)
        {
            App.Trace($"clipboard flush failed (content is copied): {ex.GetType().Name}");
        }

        SetDiagnosticsStatus("Copied.");
    }

    /// <summary>
    /// Write the report to a file the user chooses. The text written is the field the box is
    /// showing, not a fresh build, so what lands on disk is what was on screen.
    /// </summary>
    private async void OnSaveDiagnostics(object sender, RoutedEventArgs e)
    {
        try
        {
            var picker = new Windows.Storage.Pickers.FileSavePicker
            {
                SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.DocumentsLibrary,
                SuggestedFileName = $"sunno-diagnostics-{DateTime.Now:yyyy-MM-dd-HHmm}",
            };
            picker.FileTypeChoices.Add("Text file", new List<string> { ".txt" });

            // A picker created in a desktop app has no window to parent itself to and throws
            // without this.
            WinRT.Interop.InitializeWithWindow.Initialize(
                picker, WinRT.Interop.WindowNative.GetWindowHandle(this));

            var file = await picker.PickSaveFileAsync();
            if (file is null) return;   // user cancelled

            await Windows.Storage.FileIO.WriteTextAsync(file, _diagnosticsText);
            SetDiagnosticsStatus($"Saved to {file.Name}.");
        }
        catch (Exception ex)
        {
            App.Trace($"diagnostics save failed: {ex.GetType().Name}");
            SetDiagnosticsStatus($"Could not save the file ({ex.GetType().Name}). "
                                 + "Use Copy instead and paste it somewhere.");
        }
    }

    /// <summary>
    /// Delete the engine's own log files.
    ///
    /// This is a privacy action first and a housekeeping one second: backend.log is the only
    /// file on disk that can hold anything about overheard conversation, and it is deliberately
    /// excluded from the report above, so this is how someone gets rid of it.
    ///
    /// startup-trace.log is left alone. It is rewritten on every launch and is the record of
    /// the session the user is in right now, so clearing it would throw away the diagnostics
    /// they are most likely about to send.
    ///
    /// Each file is deleted on its own: the parent process appends to backend.log a line at a
    /// time, so a delete can lose a race with a write. Losing one file is not worth abandoning
    /// the rest, and the next write recreates it anyway.
    /// </summary>
    private void OnClearDiagnostics(object sender, RoutedEventArgs e)
    {
        var dir = Path.GetDirectoryName(BackendHost.LogPath);
        if (string.IsNullOrEmpty(dir)) return;

        var removed = 0;
        var failed = 0;
        foreach (var name in new[] { "backend.log", "backend.log.1", "startup-error.log" })
        {
            var path = Path.Combine(dir, name);
            try
            {
                if (!File.Exists(path)) continue;
                File.Delete(path);
                removed++;
            }
            catch (Exception ex)
            {
                failed++;
                App.Trace($"could not clear {name}: {ex.GetType().Name}");
            }
        }

        RefreshDiagnostics();
        SetDiagnosticsStatus(failed > 0
            ? $"Cleared {removed} log file(s); {failed} was in use and was left in place."
            : removed == 0
                ? "There were no log files to clear."
                : $"Cleared {removed} log file(s).");
    }

    private void OnClarityToggled(object sender, RoutedEventArgs e)
    {
        if (_settings.ShowClarity == ClarityToggle.IsOn) return;
        _settings.ShowClarity = ClarityToggle.IsOn;
        _settings.Save();
        ApplyClarityPreference();
    }

    /// <summary>
    /// Whether the clarity badge shows. Read by every caption line.
    ///
    /// Static because CaptionLine computes ShowClarity for itself and there is one window;
    /// a per-line copy of the preference would have to be pushed to every line on every change
    /// and would be wrong for any line created in between.
    /// </summary>
    private void ApplyClarityPreference()
    {
        // Fully qualified: "Models" on its own resolves to this window's model-picker
        // collection, not the namespace.
        Sunno.Models.CaptionLine.ClarityEnabled = _settings.ShowClarity;
        foreach (var line in Lines) line.RefreshClarity();
    }

    /// <summary>
    /// Move the engine between the graphics card and the processor.
    ///
    /// Reloads the model, which is the same restart the model picker already performs, so it
    /// reuses that path rather than inventing a second one. No confirmation: the model picker
    /// does not ask either, and the page says the reload takes about half a minute.
    /// </summary>
    private async void OnForceCpuToggled(object sender, RoutedEventArgs e)
    {
        if (_settings.ForceCpu == ForceCpuToggle.IsOn) return;
        _settings.ForceCpu = ForceCpuToggle.IsOn;
        _settings.Save();

        ForceCpuToggle.IsEnabled = false;
        SetDiagnosticsStatus(_settings.ForceCpu
            ? "Reloading the speech model on the processor…"
            : "Reloading the speech model on the graphics card…");

        // Same reason as the model and device switches: this deliberately replaces the engine,
        // so the app must stop treating the old one's death as the current state.
        ClearBackendFatal();

        var error = _backend.Restart(
            device: _settings.DeviceIndex?.ToString(),
            model: _settings.Model,
            vocabulary: _settings.Vocabulary,
            startStopped: _startedPaused,
            loopbackDevice: _settings.LoopbackDeviceIndex,
            computeDevice: _settings.ForceCpu ? "cpu" : "auto",
            recordingsPath: _settings.RecordingsPath,
            // Handed straight back so the restart continues the same recording instead of
            // quietly ending it and beginning another. Changing microphone mid-meeting is a
            // normal thing to do and must not cost the file.
            resumeRecording: _recording ? _activeRecordingFolder : null);

        if (!string.IsNullOrEmpty(error))
        {
            SetDiagnosticsStatus(error);
            ShowFatalBackendError(error);
        }
        else
        {
            _engineReadyThisSession = false;
            ShowLoadingState("Reloading the speech engine");
            SetDiagnosticsStatus("The speech engine is reloading. Close Settings to watch it.");
        }

        ForceCpuToggle.IsEnabled = true;
        await Task.CompletedTask;
    }
}
