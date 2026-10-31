using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Automation.Provider;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Markup;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using Microsoft.UI.Windowing;
using Sunno.Controls;
using Sunno.Services;
using Sunno.Models;
using Windows.Graphics;
using Windows.Graphics.Imaging;

namespace Sunno;

/// <summary>
/// UI-only checks of production theme resources, controller and appearance picker.
/// No BackendHost, microphone, network client, AppSettings.Load or AppSettings.Save.
/// </summary>
public partial class App : Application
{
    private Window? _window;
    private AppearanceController? _appearance;
    private int _checks;
    private int _deferredRestoreChanges;
    private SpeakerLabelsPicker _speakerPicker = null!;

    public App()
    {
        InitializeComponent();
        UnhandledException += (_, args) =>
        {
            Console.Error.WriteLine(args.Exception);
            args.Handled = true;
            Environment.Exit(1);
        };
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        _window = new Window { Title = "Sunno appearance check" };
        var root = (Grid)XamlReader.Load("""
            <Grid xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
                  Background="{ThemeResource SolidBackgroundFillColorBaseBrush}"
                  Padding="20" RowSpacing="18">
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto" />
                    <RowDefinition Height="Auto" />
                    <RowDefinition Height="*" />
                </Grid.RowDefinitions>
                <TextBlock Text="Settings" Style="{StaticResource TitleTextBlockStyle}" />
                <Border Grid.Row="1" x:Name="AppearanceCard"
                        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml" />
                <StackPanel Grid.Row="2" Spacing="12">
                    <TextBlock x:Name="CaptionPreview" xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
                               Text="Captions remain readable."
                               FontSize="26" TextWrapping="Wrap"
                               Foreground="{ThemeResource TextFillColorPrimaryBrush}" />
                    <StackPanel x:Name="Statuses" xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
                                Orientation="Horizontal" Spacing="16" />
                    <RichTextBlock x:Name="UncertainPreview" xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
                                   FontSize="26" TextWrapping="Wrap"
                                   Foreground="{ThemeResource TextFillColorPrimaryBrush}" />
                </StackPanel>
            </Grid>
            """);
        var picker = new AppearancePicker();
        picker.PreferenceChanged += (_, _) => _deferredRestoreChanges++;
        picker.Preference = AppearancePreference.Dark;
        Check(picker.Preference == AppearancePreference.Dark,
            "saved appearance is available before native control layout");
        _speakerPicker = new SpeakerLabelsPicker();
        var settingsRows = new StackPanel { Spacing = 18 };
        settingsRows.Children.Add(picker);
        settingsRows.Children.Add(_speakerPicker);
        ((Border)root.FindName("AppearanceCard")).Child = settingsRows;
        foreach (var (label, key) in new[] { ("clarity 95%", "ClarityGood"),
                                           ("clarity 65%", "ClarityMid"),
                                           ("clarity 35%", "ClarityLow") })
        {
            ((StackPanel)root.FindName("Statuses")).Children.Add(new TextBlock
            {
                Text = label, FontSize = 12, Foreground = (Brush)Resources[key],
            });
        }
        _window.Content = root;
        WordInlines.SetLine((RichTextBlock)root.FindName("UncertainPreview"), new CaptionLine
        {
            Text = "Less certain words.", IsFinal = true,
            Words = new[] { new CaptionWord("Less certain words.", 0.35) },
        });
        _window.ExtendsContentIntoTitleBar = true;
        _appearance = new AppearanceController(_window, root, AppearancePreference.System);
        var started = false;
        root.Loaded += async (_, _) =>
        {
            if (started) return;
            started = true;
            try
            {
                if (Environment.GetEnvironmentVariable("SUNNO_UPDATE_TEST_ONLY") == "1")
                    await RunUpdateChecks(root);
                else
                {
                    if (Environment.GetEnvironmentVariable("SUNNO_SCROLL_TEST_ONLY") != "1")
                        await RunChecks(root, picker);
                    await RunScrollChecks(root);
                }
                Console.WriteLine($"{_checks} appearance UI checks passed.");
                _appearance.Dispose();
                _appearance.Dispose();
                _window.Close();
                Environment.Exit(0);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex);
                _appearance.Dispose();
                _window.Close();
                Environment.Exit(1);
            }
        };
        _window.AppWindow.Resize(new SizeInt32(720, 430));
        // Show without activating, so the user's existing Sunno stays undisturbed.
        _window.AppWindow.Show(false);
    }

    private void Check(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
        _checks++;
        Console.WriteLine($"PASS {message}");
    }

    private async Task RunChecks(Grid root, AppearancePicker picker)
    {
        // Root.Loaded can precede the native RadioButtons repeater's first layout.
        await Task.Delay(100);
        Check(picker.Preference == AppearancePreference.Dark && _deferredRestoreChanges == 0,
            "native initial layout preserves the saved choice without a preference event");
        var changes = 0;
        var speakerChanges = 0;
        _speakerPicker.PreferenceChanged += (_, _) => speakerChanges++;
        var off = (RadioButton)_speakerPicker.FindName("OffOption");
        var automatic = (RadioButton)_speakerPicker.FindName("AutomaticOption");
        var speakerContainer = (Border)_speakerPicker.FindName("OptionsContainer");
        Check(!_speakerPicker.Automatic && off.IsChecked == true, "speaker labels default to Off");
        _speakerPicker.Automatic = true;
        Check(automatic.IsChecked == true && off.IsChecked == false && speakerChanges == 0,
            "saved Automatic selection restores without a user-change callback");
        ((ISelectionItemProvider)new RadioButtonAutomationPeer(off)
            .GetPattern(PatternInterface.SelectionItem)).Select();
        Check(!_speakerPicker.Automatic && speakerChanges == 1, "Off applies with one native selection");
        ((ISelectionItemProvider)new RadioButtonAutomationPeer(automatic)
            .GetPattern(PatternInterface.SelectionItem)).Select();
        Check(_speakerPicker.Automatic && off.IsChecked == false && speakerChanges == 2,
            "Automatic applies without a confirmation dialog");
        Check(new RadioButtonAutomationPeer(automatic).GetName().Contains("experimental"),
            "assistive technology exposes the experimental status");
        picker.PreferenceChanged += (_, preference) =>
        {
            changes++;
            _appearance!.Apply(preference);
        };
        picker.Preference = AppearancePreference.Dark;
        picker.Preference = AppearancePreference.System;
        Check(changes == 0, "restoring the selector does not emit a user change");
        var group = (RadioButtons)picker.FindName("ModePicker");
        var container = (Border)picker.FindName("ModeContainer");
        var label = (TextBlock)picker.FindName("ModeLabel");
        var options = group.Items.Cast<RadioButton>().ToArray();
        Check(options.Length == 3 && group.MaxColumns == 3,
            "all three appearance choices are visible without a dropdown");
        Check(group.SelectedIndex == 0 && options.Count(o => o.IsChecked == true) == 1,
            "System is the sole default selection");
        Check(group.TabNavigation == Microsoft.UI.Xaml.Input.KeyboardNavigationMode.Once,
            "native radio group provides one tab stop and arrow-key navigation");
        foreach (var (option, name) in options.Zip(new[] { "System", "Light", "Dark" }))
        {
            var peer = new RadioButtonAutomationPeer(option);
            Check(peer.GetName() == name && peer.GetAutomationControlType() == AutomationControlType.RadioButton
                  && peer.GetPattern(PatternInterface.SelectionItem) is ISelectionItemProvider,
                $"{name} exposes a named, selectable radio button to accessibility tools");
            Check(option.Content is StackPanel content && content.Children.OfType<FontIcon>().Count() == 1
                  && content.Children.OfType<TextBlock>().Single().Text == name,
                $"{name} includes a native Fluent icon and readable label");
        }
        var existingBrush = Resources["ClarityGood"];
        var uncertain = (RichTextBlock)root.FindName("UncertainPreview");
        var uncertainSpan = ((Paragraph)uncertain.Blocks[0]).Inlines.OfType<Span>().Single();
        foreach (var (preference, element, chrome, index) in new[]
                 {
                     (AppearancePreference.Light, ElementTheme.Light, TitleBarTheme.Light, 1),
                     (AppearancePreference.Dark, ElementTheme.Dark, TitleBarTheme.Dark, 2),
                 })
        {
            // Exercise the native automation selection path instead of a custom click handler.
            ((ISelectionItemProvider)new RadioButtonAutomationPeer(options[index])
                .GetPattern(PatternInterface.SelectionItem)).Select();
            await Task.Delay(200);
            Check(picker.Preference == preference && root.RequestedTheme == element
                  && root.ActualTheme == element, $"{preference} selection changes the actual theme immediately");
            Check(group.SelectedIndex == index && options.Count(o => o.IsChecked == true) == 1,
                $"{preference} is the sole selected segment");
            Check(options.Select(o => new RadioButtonAutomationPeer(o))
                    .Select(p => (ISelectionItemProvider)p.GetPattern(PatternInterface.SelectionItem))
                    .Count(p => p.IsSelected) == 1
                  && options.Select((o, i) => AutomationProperties.GetPositionInSet(o) == i + 1
                      && AutomationProperties.GetSizeOfSet(o) == 3).All(valid => valid),
                $"{preference} accessibility reports one selected choice and correct positions in the group");
            var priorChanges = changes;
            ((ISelectionItemProvider)new RadioButtonAutomationPeer(options[index])
                .GetPattern(PatternInterface.SelectionItem)).Select();
            Check(changes == priorChanges && options[index].IsChecked == true,
                $"{preference} cannot be cleared by selecting it again");
            Check(_window!.AppWindow.TitleBar.PreferredTheme == chrome,
                $"{preference} selection themes the native title-bar buttons");
            Check(ReferenceEquals(existingBrush, Resources["ClarityGood"]),
                $"{preference} switch retains brushes already bound by captions");
            var expected = AppearancePalette.ForTheme(element == ElementTheme.Dark);
            var color = ((SolidColorBrush)Resources["ClarityGood"]).Color;
            Check(color.R == (byte)(expected.Good >> 16) && color.G == (byte)(expected.Good >> 8)
                  && color.B == (byte)expected.Good, $"{preference} status text has the matching palette");
            var caption = ((SolidColorBrush)((TextBlock)root.FindName("CaptionPreview")).Foreground).Color;
            Check(element == ElementTheme.Dark ? caption.R > 128 : caption.R < 128,
                $"{preference} theme resources recolor existing caption text");
            Check(ReferenceEquals(uncertainSpan.Foreground, Resources["CaptionUncertain"]),
                $"{preference} switch preserves the brush on already-rendered uncertain words");
            var uncertainColor = ((SolidColorBrush)uncertainSpan.Foreground).Color;
            Check(element == ElementTheme.Dark ? uncertainColor.R > 128 : uncertainColor.R < 128,
                $"{preference} switch recolors existing uncertain caption words");

            foreach (var width in new[] { 720, 354 })
            {
                var scale = root.XamlRoot.RasterizationScale;
                _window.AppWindow.Resize(new SizeInt32((int)(width * scale), (int)(430 * scale)));
                await Task.Delay(150);
                Check(picker.ActualWidth >= container.ActualWidth
                      && options.All(o => o.ActualWidth >= 78 && o.ActualHeight >= 32),
                    $"{preference} segments fit at {width}px, including the minimum settings-card width");
                var origins = options.Select(o => o.TransformToVisual(group)
                    .TransformPoint(new Windows.Foundation.Point())).ToArray();
                Check(options.All(o => Math.Abs(o.ActualWidth - options[0].ActualWidth) < 1),
                    $"{preference} segments are equal-width at {width}px");
                Check(origins.All(p => Math.Abs(p.Y - origins[0].Y) < 1)
                      && origins.Zip(options).All(pair => pair.First.X + pair.Second.ActualWidth <= group.ActualWidth + 1),
                    $"{preference} keeps every segment unclipped on one row at {width}px");
                Check(width < 420 ? Grid.GetRow(container) == 1 && Grid.GetColumnSpan(container) == 2
                                 : Grid.GetRow(container) == 0 && Grid.GetColumn(container) == 1
                                   && label.ActualWidth + container.ActualWidth + 16 <= picker.ActualWidth + 1,
                    $"{preference} adapts label placement at {width}px");
                await Capture(root, $"{preference}-{width}.png");
                Check(speakerContainer.ActualWidth <= _speakerPicker.ActualWidth &&
                      off.ActualHeight >= 32 && automatic.ActualHeight >= 32,
                    $"speaker controls fit without clipping in {preference} at {width}px");
                Check(width < 380 ? Grid.GetRow(speakerContainer) == 1
                                 : Grid.GetRow(speakerContainer) == 0,
                    $"speaker setting reflows in {preference} at {width}px");
            }
            var flyout = new MenuFlyout();
            flyout.Items.Add(new MenuFlyoutItem { Text = "Theme check" });
            flyout.ShowAt(picker);
            await Task.Delay(100);
            var popupThemes = VisualTreeHelper.GetOpenPopupsForXamlRoot(root.XamlRoot)
                .Select(p => p.Child).OfType<FrameworkElement>().ToArray();
            Check(popupThemes.Length > 0 && popupThemes.All(p => p.ActualTheme == element),
                $"{preference} menus inherit the owning control's theme");
            flyout.Hide();
            var dialog = new ContentDialog
            {
                Title = "Theme check", Content = "Dialogs match the selected appearance.",
                CloseButtonText = "Close", XamlRoot = root.XamlRoot,
                RequestedTheme = root.RequestedTheme,
            };
            var shown = dialog.ShowAsync();
            await Task.Delay(150);
            Check(dialog.ActualTheme == element, $"{preference} dialogs respect the root override");
            dialog.Hide();
            await shown;
        }
        ((ISelectionItemProvider)new RadioButtonAutomationPeer(options[0])
            .GetPattern(PatternInterface.SelectionItem)).Select();
        await Task.Delay(200);
        Check(changes == 3, "each real selection raises one preference event");
        Check(root.RequestedTheme == ElementTheme.Default,
            "System removes the explicit light/dark override");
        var systemTheme = RequestedTheme == ApplicationTheme.Dark ? ElementTheme.Dark : ElementTheme.Light;
        Check(root.ActualTheme == systemTheme, "System resolves to the Windows app appearance");
        Check(_window!.AppWindow.TitleBar.PreferredTheme == TitleBarTheme.UseDefaultAppMode,
            "System also restores Windows app-mode title-bar theming");
        var changesBeforeRestore = changes;
        picker.Preference = " LIGHT ";
        Check(picker.Preference == AppearancePreference.Light && group.SelectedIndex == 1
              && changes == changesBeforeRestore, "restoring Light normalizes the saved value without emitting an event");
        picker.Preference = "unknown";
        Check(picker.Preference == AppearancePreference.System && group.SelectedIndex == 0
              && changes == changesBeforeRestore, "invalid saved appearance safely restores System without emitting an event");
        _appearance!.Dispose();
        _appearance.Apply(AppearancePreference.Dark);
        Check(root.RequestedTheme == ElementTheme.Default, "disposed controllers do not update a closed window");
    }

    private async Task RunScrollChecks(Grid root)
    {
        // Real WinUI layout and scrolling, without running Sunno or capturing audio.
        root.Children.Clear();
        root.RowDefinitions.Clear();
        root.Padding = new Thickness(0);
        var rows = new StackPanel();
        var host = new CaptionScrollViewer
        {
            ScrollContent = rows, ScrollPadding = new Thickness(24, 40, 24, 16),
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled, VerticalAnchorRatio = 0,
        };
        var viewer = host.Viewer;
        root.Children.Add(host);
        _window!.AppWindow.Resize(new SizeInt32(720, 430));
        var animationsEnabled = true;
        using var follow = new CaptionScrollFollower(host, rows, () => animationsEnabled);
        var skip = new SkipToCurrentButton
        {
            HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Bottom,
            Margin = new Thickness(0, 0, 0, 12), Visibility = Visibility.Collapsed,
        };
        root.Children.Add(skip);
        follow.Changed += () => skip.Visibility = follow.CanSkipToCurrent ? Visibility.Visible : Visibility.Collapsed;
        skip.Click += (_, _) => { viewer.Focus(FocusState.Programmatic); follow.Resume(animate: true); };
        TextBlock Add(string text = "A caption in the live transcript.")
        {
            var row = new TextBlock { Text = text, FontSize = 26, TextWrapping = TextWrapping.Wrap,
                Margin = new Thickness(0, 0, 0, 14), CanBeScrollAnchor = true };
            rows.Children.Add(row);
            follow.RequestFollow();
            return row;
        }
        for (var i = 0; i < 60; i++) Add($"Caption {i}: words arrive here.");
        await Task.Delay(250);
        Check(follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 1,
            "new captions initially follow the measured live edge");
        Check(skip.Visibility == Visibility.Collapsed, "skip control stays hidden while following");
        // Exercise the asynchronous movement that previously looked like a user scroll.
        viewer.ChangeView(null, viewer.VerticalOffset - 35, null, disableAnimation: true);
        await Task.Delay(180);
        Check(follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 1,
            "an automatic offset adjustment cannot turn following off");
        for (var turn = 0; turn < 12; turn++)
        {
            var line = new CaptionLine { Text = "The provisional words become a finished caption when speech stops." };
            var block = new RichTextBlock { FontSize = 26, TextWrapping = TextWrapping.Wrap,
                Margin = new Thickness(0, 0, 0, 14), CanBeScrollAnchor = true };
            WordInlines.SetLine(block, line);
            rows.Children.Add(block);
            follow.RequestFollow();
            await Task.Delay(80);
            line.Text = "The finished words become a caption when speech stops.";
            line.Words = line.Text.Split(' ').Select(word => new CaptionWord(" " + word, .95)).ToArray();
            line.IsFinal = true;
            follow.RequestFollow();
            await Task.Delay(80);
            Check(follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 1,
                $"final word rendering preserves following on turn {turn} (following={follow.IsFollowing}, gap={viewer.ScrollableHeight - viewer.VerticalOffset:0.0})");
        }
        var revised = Add(string.Join(" ", Enumerable.Repeat("A provisional caption before the video pauses.", 20)));
        await Task.Delay(200);
        revised.Text = "A shorter final caption after the video pauses.";
        follow.RequestFollow();
        await Task.Delay(200);
        Add("The video resumes and new words arrive.");
        await Task.Delay(200);
        Check(follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 1,
            $"following survives a shortened final caption when video resumes (following={follow.IsFollowing}, gap={viewer.ScrollableHeight - viewer.VerticalOffset:0.0})");
        // Production keeps 200 rows. Replacing equally tall rows moves WinUI's anchor
        // without changing ScrollableHeight, so geometry alone cannot identify a gesture.
        while (rows.Children.Count < 200) Add($"Caption {rows.Children.Count}: words arrive here.");
        await Task.Delay(200);
        for (var i = 0; i < 12; i++)
        {
            rows.Children.RemoveAt(0);
            Add($"Caption {200 + i}: words arrive here.");
            await Task.Delay(60);
        }
        Check(follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 1,
            $"following survives rolling history at the 200-caption limit (following={follow.IsFollowing}, gap={viewer.ScrollableHeight - viewer.VerticalOffset:0.0})");
        var scroll = (IScrollProvider)FrameworkElementAutomationPeer.CreatePeerForElement(host).GetPattern(PatternInterface.Scroll);
        scroll.SetScrollPercent(-1, 100 * (viewer.ScrollableHeight - 40) / viewer.ScrollableHeight);
        await Task.Delay(150);
        Check(!follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 80,
            "even a short upward scroll within the near-bottom zone pauses following");
        var shortScrollOffset = viewer.VerticalOffset;
        Add("A caption arriving after a small upward scroll.");
        await Task.Delay(150);
        Check(!follow.IsFollowing && Math.Abs(viewer.VerticalOffset - shortScrollOffset) < 1,
            "incoming captions do not undo a small upward scroll");
        scroll.SetScrollPercent(-1, 100);
        await Task.Delay(150);
        Check(follow.IsFollowing, "a downward return resumes after a short upward scroll");
        scroll.Scroll(ScrollAmount.NoAmount, ScrollAmount.LargeDecrement);
        await Task.Delay(200);
        Check(!follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset > 80,
            "native page-up scrolling pauses following");
        var readingOffset = viewer.VerticalOffset;
        var partial = Add("A new partial caption.");
        await Task.Delay(150);
        Check(!follow.IsFollowing && Math.Abs(viewer.VerticalOffset - readingOffset) < 1,
            "new captions leave the reading position alone");
        partial.Text = string.Join(" ", Enumerable.Repeat("More words in this partial caption.", 30));
        follow.RequestFollow();
        await Task.Delay(150);
        Check(!follow.IsFollowing && Math.Abs(viewer.VerticalOffset - readingOffset) < 1,
            "a growing partial does not resume following");
        var skipButton = (Button)skip.FindName("SkipButton");
        Check(skip.Visibility == Visibility.Visible && AutomationProperties.GetName(skipButton) == "Skip to current",
            "skip control is visible and named for accessibility while reading history");
        foreach (var theme in new[] { ElementTheme.Light, ElementTheme.Dark })
        {
            root.RequestedTheme = theme;
            await Task.Delay(100);
            await Capture(root, $"scroll-{theme.ToString().ToLowerInvariant()}.png");
            var presenter = (ContentPresenter)VisualTreeHelper.GetChild(skipButton, 0);
            var normal = ((SolidColorBrush)presenter.Background).Color;
            var stableHover = true;
            for (var hover = 0; hover < 6; hover++)
            {
                VisualStateManager.GoToState(skipButton, hover % 2 == 0 ? "PointerOver" : "Normal", true);
                await Task.Delay(25);
                var color = ((SolidColorBrush)presenter.Background).Color;
                stableHover &= Math.Abs(color.R - normal.R) <= 20 && color.A >= 230;
            }
            Check(stableHover, $"{theme} rapid hover keeps the pill's translucent colors without a flash");
            VisualStateManager.GoToState(skipButton, "PointerOver", true);
            await Task.Delay(50);
            await Capture(root, $"scroll-{theme.ToString().ToLowerInvariant()}-hover.png");
            VisualStateManager.GoToState(skipButton, "Normal", true);
        }
        _window.AppWindow.Resize(new SizeInt32(390, 260));
        await Task.Delay(150);
        Check(skip.ActualWidth <= root.ActualWidth && skip.ActualHeight >= 32,
            "skip pill fits a compact caption window");
        VisualStateManager.GoToState(skipButton, "PointerOver", false);
        await Capture(root, "scroll-compact-hover.png");
        VisualStateManager.GoToState(skipButton, "Normal", false);
        _window.AppWindow.Resize(new SizeInt32(720, 430));
        await Task.Delay(150);
        // Resizing can reflow text while paused; take the reading anchor after it settles.
        readingOffset = viewer.VerticalOffset;
        var anchor = rows.Children.OfType<TextBlock>().First(row =>
            row.TransformToVisual(viewer).TransformPoint(new Windows.Foundation.Point()).Y >= 0);
        var anchorY = anchor.TransformToVisual(viewer).TransformPoint(new Windows.Foundation.Point()).Y;
        rows.Children[0].Visibility = Visibility.Collapsed;
        await Task.Delay(150);
        Check(!follow.IsFollowing && Math.Abs(anchor.TransformToVisual(viewer)
                .TransformPoint(new Windows.Foundation.Point()).Y - anchorY) < 1,
            "metadata layout changes above the viewport preserve the reading anchor");
        for (var i = 0; i < 10; i++) rows.Children.RemoveAt(0);
        Add("Another caption after old history is trimmed.");
        await Task.Delay(150);
        Check(!follow.IsFollowing && Math.Abs(anchor.TransformToVisual(viewer)
                .TransformPoint(new Windows.Foundation.Point()).Y - anchorY) < 1,
            "trimming old history keeps the same visible caption anchored");
        // A native scroll, not an application follow request, resumes at the live edge.
        scroll.SetScrollPercent(-1, 100 * (viewer.ScrollableHeight - 40) / viewer.ScrollableHeight);
        await Task.Delay(200);
        Check(follow.IsFollowing, "scrolling within 80 DIPs of the bottom resumes following");
        Add("A new final caption after following resumes.");
        await Task.Delay(150);
        Check(viewer.ScrollableHeight - viewer.VerticalOffset < 1,
            "captions follow again after returning near the bottom");
        // Queue a follow, then scroll away before that low-priority callback executes.
        Add("Queued caption.");
        scroll.SetScrollPercent(-1, 20);
        await Task.Delay(200);
        Check(!follow.IsFollowing && viewer.VerticalOffset < viewer.ScrollableHeight / 2,
            "a queued follow cannot override a newer scroll away");
        _window.AppWindow.Resize(new SizeInt32(620, 540));
        await Task.Delay(150);
        Check(!follow.IsFollowing, "resizing while reading history does not turn following back on");
        var invokeSkip = (IInvokeProvider)FrameworkElementAutomationPeer.CreatePeerForElement(skipButton)
            .GetPattern(PatternInterface.Invoke);
        Check(CaptionScrollFollower.JumpDurationSeconds(20, 500) == .2
              && CaptionScrollFollower.JumpDurationSeconds(2500, 500) > .2
              && CaptionScrollFollower.JumpDurationSeconds(100000, 500) == 1,
            "jump duration scales with distance and stays between 0.2 and 1 second");
        var jumpClock = System.Diagnostics.Stopwatch.StartNew();
        invokeSkip.Invoke();
        await Task.Delay(80);
        Check(follow.IsJumping && viewer.ScrollableHeight - viewer.VerticalOffset > 1,
            "Skip to current animates instead of jumping instantly");
        Add("New captions continue arriving during the jump.");
        while (follow.IsJumping && jumpClock.ElapsedMilliseconds < 1400) await Task.Delay(20);
        await Task.Delay(60);
        Check(follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 1
              && skip.Visibility == Visibility.Collapsed && jumpClock.ElapsedMilliseconds <= 1400,
            "animated skip reaches the moving live edge within its bounded duration");
        scroll.SetScrollPercent(-1, 0);
        await Task.Delay(100);
        follow.Resume(animate: true);
        await Task.Delay(80);
        scroll.Scroll(ScrollAmount.NoAmount, ScrollAmount.LargeDecrement);
        await Task.Delay(120);
        Check(!follow.IsJumping && !follow.IsFollowing, "reader navigation interrupts the animated jump");
        var interruptedOffset = viewer.VerticalOffset;
        await Task.Delay(1000);
        Check(Math.Abs(viewer.VerticalOffset - interruptedOffset) < 1, "an interrupted jump never resumes behind the reader");
        animationsEnabled = false;
        follow.Resume(animate: true);
        await Task.Delay(100);
        Check(!follow.IsJumping && follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 1,
            "reduced motion skips the spatial animation and resumes following");
        animationsEnabled = true;
        Add("New words after skipping to current.");
        await Task.Delay(100);
        rows.Children.RemoveAt(rows.Children.Count - 1);
        await Task.Delay(150);
        Add("Captions resume after a discarded provisional line.");
        await Task.Delay(150);
        Check(follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 1,
            "following survives discarded provisional captions and resumed speech");
        rows.Children.Clear();
        follow.Reset();
        for (var i = 0; i < 30; i++) Add("A fresh transcript.");
        await Task.Delay(200);
        Check(follow.IsFollowing && viewer.ScrollableHeight - viewer.VerticalOffset < 1,
            "clearing the transcript restores live following");
        follow.Dispose();
        follow.Dispose();
        Check(true, "scroll-follow subscriptions dispose idempotently");
    }

    private async Task RunUpdateChecks(Grid root)
    {
        root.Children.Clear(); root.RowDefinitions.Clear(); root.Padding = new Thickness(24);
        _window!.AppWindow.Resize(new SizeInt32(380, 440));
        var about = new StackPanel { Spacing = 10, Width = 300, HorizontalAlignment = HorizontalAlignment.Left };
        about.Children.Add(new TextBlock { Text = "About this app", Style = (Style)Resources["BodyStrongTextBlockStyle"] });
        about.Children.Add((TextBlock)XamlReader.Load("""
            <TextBlock xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
                Text="Sunno 1.0.90.0" Foreground="{ThemeResource TextFillColorSecondaryBrush}" />
            """));
        var panel = new AppUpdatePanel { Margin = new Thickness(0, 6, 0, 10) };
        about.Children.Add(panel);
        about.Children.Add(new HyperlinkButton { Content = "Privacy policy", Padding = new Thickness(0) });
        about.Children.Add(new HyperlinkButton { Content = "Source code", Padding = new Thickness(0) });
        root.Children.Add(about);
        var action = (Button)panel.FindName("UpdateAction");
        var cancel = (Button)panel.FindName("CancelAction");
        var more = (Button)panel.FindName("UpdateOptions");
        var progress = (ProgressBar)panel.FindName("DownloadProgress");
        var calls = 0;
        panel.ActionRequested += (_, _) => calls++;
        panel.CancelRequested += (_, _) => calls++;
        foreach (var theme in new[] { ElementTheme.Light, ElementTheme.Dark })
        {
            root.RequestedTheme = theme;
            panel.Show("Check for updates");
            await Task.Delay(100);
            Check(action.IsEnabled && progress.Visibility == Visibility.Collapsed, $"{theme} update check starts idle without background activity");
            await Capture(root, $"updates-{theme}-idle.png");
            panel.Show("Downloading...", "Downloading Sunno 1.0.91.0: 42%", busy: true, progress: 42);
            await Task.Delay(80);
            Check(!action.IsEnabled && !more.IsEnabled && cancel.Visibility == Visibility.Visible && progress.Value == 42,
                $"{theme} download prevents duplicate actions and permits cancellation");
            Check(action.ActualWidth + cancel.ActualWidth + more.ActualWidth + 12 <= panel.ActualWidth,
                $"{theme} update controls fit the actual 300-DIP Settings column");
            await Capture(root, $"updates-{theme}-download.png");
            panel.Show("Install update", "Sunno 1.0.91.0 is ready. Updating closes Sunno and clears the current live transcript.");
            await Task.Delay(80);
            Check(action.IsEnabled && cancel.Visibility == Visibility.Collapsed && progress.Visibility == Visibility.Collapsed,
                $"{theme} verified download offers installation");
            await Capture(root, $"updates-{theme}-ready.png");
            panel.Show("Check for updates", "Could not reach GitHub. Check your connection and try again.");
            await Task.Delay(80);
            Check(action.IsEnabled && more.IsEnabled, $"{theme} failure retains retry and release-page access");
            await Capture(root, $"updates-{theme}-error.png");
        }
        ((IInvokeProvider)FrameworkElementAutomationPeer.CreatePeerForElement(action).GetPattern(PatternInterface.Invoke)).Invoke();
        panel.Show("Checking...", "Checking GitHub releases...", busy: true);
        await Task.Delay(80);
        ((IInvokeProvider)FrameworkElementAutomationPeer.CreatePeerForElement(cancel).GetPattern(PatternInterface.Invoke)).Invoke();
        await Task.Delay(80);
        Check(calls == 2, "native check and cancel actions reach their handlers exactly once");
        Check(AutomationProperties.GetLiveSetting((TextBlock)panel.FindName("UpdateStatus")) == AutomationLiveSetting.Polite,
            "update status is announced politely to assistive technology");
    }

    private static async Task Capture(FrameworkElement root, string name)
    {
        var output = Environment.GetEnvironmentVariable("SUNNO_APPEARANCE_TEST_OUTPUT");
        if (string.IsNullOrWhiteSpace(output)) return;
        Directory.CreateDirectory(output);
        var bitmap = new RenderTargetBitmap();
        await bitmap.RenderAsync(root);
        var pixels = (await bitmap.GetPixelsAsync()).ToArray();
        using var file = File.Create(Path.Combine(output, name));
        using var stream = file.AsRandomAccessStream();
        var encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.PngEncoderId, stream);
        encoder.SetPixelData(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied,
            (uint)bitmap.PixelWidth, (uint)bitmap.PixelHeight, 96, 96, pixels);
        await encoder.FlushAsync();
    }
}
