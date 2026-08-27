using System;
using System.Diagnostics;
using System.IO;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Animation;
using Sunno.Services;

namespace Sunno;

/// <summary>
/// The record control and everything that drives it.
///
/// Split out of MainWindow.xaml.cs, which is already past three thousand lines. The states
/// are a small machine and they read better in one place than threaded through a file that
/// is mostly about captions.
///
/// idle -> recording -> saving -> saved -> idle. The pill grows leftward out of a 30px
/// circle, carries a timer while running, shrinks back for the spinner and the tick, and
/// then returns to rest. Nothing here writes files; the backend owns the recording and this
/// only reflects what it reports, so a state that is showing is one the backend has actually
/// reached.
/// </summary>
public sealed partial class MainWindow
{
    private DispatcherQueueTimer? _recordTimer;
    private DispatcherQueueTimer? _savedHold;
    private Storyboard? _pulse;
    private double _recordElapsed;
    private string? _lastSavedFolder;
    private bool _recording;

    /// <summary>Where the backend writes recordings. Empty means its own default.</summary>
    private string RecordingsPath => _settings.RecordingsPath ?? "";

    private async void OnToggleRecording(object sender, RoutedEventArgs e)
    {
        // Guarded on the backend too, which is what actually decides. This only stops a
        // second click landing while the first is in flight.
        if (!_client.IsConnected)
        {
            ShowDeviceNotice("Sunno is still starting up.");
            return;
        }

        if (_recording)
        {
            await _client.StopRecordingAsync();
            return;
        }

        RecordButton.IsEnabled = false;
        try
        {
            await _client.StartRecordingAsync(
                string.IsNullOrWhiteSpace(RecordingsPath) ? null : RecordingsPath);
        }
        finally
        {
            RecordButton.IsEnabled = true;
        }
    }

    /// <summary>
    /// Say something once through the transcript's live region.
    ///
    /// Reuses CaptionAnnouncer rather than adding a second live region: two of them compete,
    /// and a screen reader given both will interleave a status message into the middle of a
    /// sentence somebody is speaking.
    /// </summary>
    private void AnnounceRecording(string message)
    {
        try
        {
            CaptionAnnouncer.Text = message;
            var peer = FrameworkElementAutomationPeer.CreatePeerForElement(CaptionAnnouncer);
            peer?.RaiseAutomationEvent(AutomationEvents.LiveRegionChanged);
        }
        catch (Exception ex)
        {
            App.Trace($"recording announce failed: {ex.Message}");
        }
    }

    /// <summary>Reflect what the backend says it is doing.</summary>
    private void OnRecordingState(RecordingState state)
    {
        // Wrapped whole. The recording itself is already safe on disk by the time any of
        // this runs, so a fault in the animations must not be allowed to take the window
        // down: losing the app at the exact moment somebody is watching for confirmation
        // that their meeting was saved is the worst possible time for it. An earlier draft
        // animated UIElement.Scale from a Storyboard, which is a composition property, and
        // did precisely that.
        try
        {
            ApplyRecordingState(state);
        }
        catch (Exception ex)
        {
            App.Trace($"recording state '{state.State}' failed to render: {ex}");
            try
            {
                EnterIdle();
            }
            catch (Exception inner)
            {
                App.Trace($"recording idle fallback failed: {inner.Message}");
            }
        }
    }

    private void ApplyRecordingState(RecordingState state)
    {
        switch (state.State)
        {
            case "recording":
                _recordElapsed = state.ElapsedSeconds;
                EnterRecording();
                break;
            case "saving":
                EnterSaving();
                break;
            case "saved":
                _lastSavedFolder = state.Folder;
                EnterSaved(state);
                break;
            case "failed":
                EnterIdle();
                ShowDeviceNotice(string.IsNullOrEmpty(state.Message)
                    ? "Could not save the recording."
                    : $"Could not save the recording. {state.Message}");
                break;
            default:
                EnterIdle();
                break;
        }
    }

    private void EnterRecording()
    {
        if (_recording) return;
        _recording = true;
        StopSavedHold();

        RecordIdleMark.Visibility = Visibility.Collapsed;
        RecordSavingRing.Visibility = Visibility.Collapsed;
        RecordSavingRing.IsActive = false;
        RecordSavedTick.Visibility = Visibility.Collapsed;
        RecordLiveDot.Visibility = Visibility.Visible;
        RecordElapsed.Visibility = Visibility.Visible;
        RecordElapsed.Text = FormatRecordElapsed(_recordElapsed);

        AutomationProperties.SetName(RecordButton, "Stop recording");
        ToolTipService.SetToolTip(RecordButton, "Stop and save this recording");

        FadePill(1.0);
        GrowPill();
        StartPulse();

        _recordTimer ??= _ui.CreateTimer();
        _recordTimer.Interval = TimeSpan.FromSeconds(1);
        _recordTimer.Tick -= OnRecordTick;
        _recordTimer.Tick += OnRecordTick;
        _recordTimer.Start();

        // Said once, out loud, because the pill is a small target in the corner and someone
        // using a screen reader should not have to go looking for it to know.
        AnnounceRecording("Recording. Saving to your recordings folder.");
    }

    private void OnRecordTick(DispatcherQueueTimer sender, object args)
    {
        _recordElapsed += 1;
        RecordElapsed.Text = FormatRecordElapsed(_recordElapsed);
        // Re-measure as the digits grow: 9:59 to 10:00 is a wider string, and a pill that
        // stayed at its old width would clip it.
        GrowPill();
    }

    private void EnterSaving()
    {
        _recording = false;
        StopPulse();
        _recordTimer?.Stop();

        RecordLiveDot.Visibility = Visibility.Collapsed;
        RecordElapsed.Visibility = Visibility.Collapsed;
        RecordIdleMark.Visibility = Visibility.Collapsed;
        RecordSavedTick.Visibility = Visibility.Collapsed;
        RecordSavingRing.Visibility = Visibility.Visible;
        RecordSavingRing.IsActive = true;

        // No label. The state lasts about a second, and a word would be read after the
        // moment it describes had already passed.
        ShrinkPill();
        AutomationProperties.SetName(RecordButton, "Saving recording");
    }

    private void EnterSaved(RecordingState state)
    {
        _recording = false;
        StopPulse();
        _recordTimer?.Stop();

        RecordSavingRing.IsActive = false;
        RecordSavingRing.Visibility = Visibility.Collapsed;
        RecordLiveDot.Visibility = Visibility.Collapsed;
        RecordElapsed.Visibility = Visibility.Collapsed;
        RecordIdleMark.Visibility = Visibility.Collapsed;
        RecordSavedTick.Visibility = Visibility.Visible;
        RecordSavedTick.Opacity = 0;
        // RenderTransform, not UIElement.Scale. Scale is a composition property and a XAML
        // Storyboard cannot target it: doing so throws a stowed exception that takes the
        // whole app down, which is exactly the wrong moment for that to happen because the
        // recording has just finished and the user is watching for confirmation.
        var scale = new ScaleTransform { ScaleX = 0.4, ScaleY = 0.4, CenterX = 7, CenterY = 7 };
        RecordSavedTick.RenderTransform = scale;

        FadePill(1.0);
        ShrinkPill();
        DrawTick();

        AutomationProperties.SetName(RecordButton, "Recording saved");
        ToolTipService.SetToolTip(RecordButton,
            $"Saved {state.Name} ({FormatRecordElapsed(state.DurationSeconds)})");
        AnnounceRecording($"Saved {state.Name}, {FormatRecordElapsed(state.DurationSeconds)}.");

        // Held long enough to be seen and short enough not to become the resting state. The
        // tick is the only confirmation there is, so it must not be missable, and it must
        // not linger as though something still needs attention.
        _savedHold ??= _ui.CreateTimer();
        _savedHold.Interval = TimeSpan.FromSeconds(2.2);
        _savedHold.Tick -= OnSavedHoldElapsed;
        _savedHold.Tick += OnSavedHoldElapsed;
        _savedHold.Start();
    }

    private void OnSavedHoldElapsed(DispatcherQueueTimer sender, object args)
    {
        sender.Stop();
        EnterIdle();
    }

    private void StopSavedHold()
    {
        _savedHold?.Stop();
    }

    private void EnterIdle()
    {
        _recording = false;
        StopPulse();
        StopSavedHold();
        _recordTimer?.Stop();
        _recordElapsed = 0;

        RecordSavingRing.IsActive = false;
        RecordSavingRing.Visibility = Visibility.Collapsed;
        RecordSavedTick.Visibility = Visibility.Collapsed;
        RecordLiveDot.Visibility = Visibility.Collapsed;
        RecordElapsed.Visibility = Visibility.Collapsed;
        RecordIdleMark.Visibility = Visibility.Visible;
        RecordIdleMark.Opacity = 1;

        FadePill(0.0);
        ShrinkPill();
        AutomationProperties.SetName(RecordButton, "Start recording");
        ToolTipService.SetToolTip(RecordButton, "Save this conversation to a file");
    }

    // ---- animation -------------------------------------------------------------------

    /// <summary>Width the pill needs for its current contents.</summary>
    private double DesiredPillWidth()
    {
        if (!_recording) return 30;
        RecordElapsed.Measure(new Windows.Foundation.Size(
            double.PositiveInfinity, double.PositiveInfinity));
        // text + left inset + gap + dot + right inset
        return Math.Max(30, RecordElapsed.DesiredSize.Width + 16 + 8 + 10 + 9);
    }

    private void GrowPill() => AnimateWidth(DesiredPillWidth());

    private void ShrinkPill() => AnimateWidth(30);

    private void AnimateWidth(double target)
    {
        if (Math.Abs(RecordPill.Width - target) < 0.5 && RecordPill.Width > 0) return;

        var from = double.IsNaN(RecordPill.Width) ? 30 : RecordPill.Width;
        var anim = new DoubleAnimation
        {
            From = from,
            To = target,
            Duration = new Duration(TimeSpan.FromMilliseconds(260)),
            // WinUI cannot animate Width on the composition thread, so EnableDependent-
            // Animation is required. It is one property on a 30px element in the corner of
            // the window; the cost is not measurable and the alternative is a jump.
            EnableDependentAnimation = true,
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut },
        };
        Storyboard.SetTarget(anim, RecordPill);
        Storyboard.SetTargetProperty(anim, "Width");
        var sb = new Storyboard();
        sb.Children.Add(anim);
        sb.Begin();

        RecordVisual.MinWidth = target;
    }

    private void FadePill(double opacity)
    {
        var anim = new DoubleAnimation
        {
            To = opacity,
            Duration = new Duration(TimeSpan.FromMilliseconds(200)),
        };
        Storyboard.SetTarget(anim, RecordPill);
        Storyboard.SetTargetProperty(anim, "Opacity");
        var sb = new Storyboard();
        sb.Children.Add(anim);
        sb.Begin();
    }

    /// <summary>The recording dot, breathing rather than blinking.</summary>
    private void StartPulse()
    {
        StopPulse();
        var anim = new DoubleAnimation
        {
            From = 1.0,
            To = 0.35,
            Duration = new Duration(TimeSpan.FromMilliseconds(900)),
            AutoReverse = true,
            RepeatBehavior = RepeatBehavior.Forever,
            EasingFunction = new SineEase { EasingMode = EasingMode.EaseInOut },
        };
        Storyboard.SetTarget(anim, RecordLiveDot);
        Storyboard.SetTargetProperty(anim, "Opacity");
        _pulse = new Storyboard();
        _pulse.Children.Add(anim);
        _pulse.Begin();
    }

    private void StopPulse()
    {
        _pulse?.Stop();
        _pulse = null;
        RecordLiveDot.Opacity = 1;
    }

    /// <summary>The tick, drawn on rather than appearing.</summary>
    private void DrawTick()
    {
        if (RecordSavedTick.RenderTransform is not ScaleTransform scale) return;

        var sb = new Storyboard();

        var fade = new DoubleAnimation
        {
            From = 0,
            To = 1,
            Duration = new Duration(TimeSpan.FromMilliseconds(180)),
        };
        Storyboard.SetTarget(fade, RecordSavedTick);
        Storyboard.SetTargetProperty(fade, "Opacity");
        sb.Children.Add(fade);

        // A small overshoot, so it lands rather than fades in. This is the only feedback the
        // user gets that a meeting is now a file, and it should feel like a result.
        foreach (var axis in new[] { "ScaleX", "ScaleY" })
        {
            var pop = new DoubleAnimationUsingKeyFrames { EnableDependentAnimation = true };
            pop.KeyFrames.Add(new EasingDoubleKeyFrame
            { KeyTime = TimeSpan.Zero, Value = 0.4 });
            pop.KeyFrames.Add(new EasingDoubleKeyFrame
            {
                KeyTime = TimeSpan.FromMilliseconds(200),
                Value = 1.18,
                EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut },
            });
            pop.KeyFrames.Add(new EasingDoubleKeyFrame
            {
                KeyTime = TimeSpan.FromMilliseconds(320),
                Value = 1.0,
                EasingFunction = new CubicEase { EasingMode = EasingMode.EaseInOut },
            });
            Storyboard.SetTarget(pop, scale);
            Storyboard.SetTargetProperty(pop, axis);
            sb.Children.Add(pop);
        }

        sb.Begin();
    }

    private static string FormatRecordElapsed(double seconds)
    {
        var t = TimeSpan.FromSeconds(Math.Max(0, seconds));
        return t.TotalHours >= 1
            ? $"{(int)t.TotalHours}:{t.Minutes:00}:{t.Seconds:00}"
            : $"{t.Minutes:00}:{t.Seconds:00}";
    }

    // ---- settings --------------------------------------------------------------------

    private string EffectiveRecordingsPath()
    {
        if (!string.IsNullOrWhiteSpace(RecordingsPath)) return RecordingsPath;
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "Sunno", "Recordings");
    }

    /// <summary>Show the current destination on the Settings page.</summary>
    private void RefreshRecordingsPath()
    {
        if (RecordingsPathBox is not null)
            RecordingsPathBox.Text = EffectiveRecordingsPath();
    }

    private async void OnChangeRecordingsFolder(object sender, RoutedEventArgs e)
    {
        var picker = new Windows.Storage.Pickers.FolderPicker();
        // WinUI 3 pickers are window-owned and throw without a handle. There is no Window
        // property on a picker, so the interop initialiser is the only way.
        WinRT.Interop.InitializeWithWindow.Initialize(
            picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
        picker.FileTypeFilter.Add("*");
        picker.SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.DocumentsLibrary;

        try
        {
            var folder = await picker.PickSingleFolderAsync();
            if (folder is null) return;

            _settings.RecordingsPath = folder.Path;
            _settings.Save();
            RefreshRecordingsPath();

            // Takes effect for the next recording rather than the one running. Moving the
            // destination out from under an open file would leave half a meeting in one
            // folder and half in another.
            if (_recording)
                ShowDeviceNotice("The new folder will be used for your next recording.");
        }
        catch (Exception ex)
        {
            App.Trace($"folder picker failed: {ex.Message}");
            ShowDeviceNotice("Could not open the folder picker.");
        }
    }

    private void OnOpenRecordingsFolder(object sender, RoutedEventArgs e)
        => OpenRecordingsFolder();

    /// <summary>Open the recordings folder, or the last saved recording inside it.</summary>
    private void OpenRecordingsFolder(bool lastSaved = false)
    {
        var target = lastSaved && !string.IsNullOrEmpty(_lastSavedFolder)
            ? _lastSavedFolder!
            : EffectiveRecordingsPath();
        try
        {
            // Created on demand rather than at startup: an install that never records should
            // leave nothing behind, so this is the first moment the folder has to exist.
            Directory.CreateDirectory(target);
            Process.Start(new ProcessStartInfo(target) { UseShellExecute = true });
        }
        catch (Exception ex)
        {
            App.Trace($"open recordings folder failed: {ex.Message}");
            ShowDeviceNotice("Could not open that folder.");
        }
    }
}
