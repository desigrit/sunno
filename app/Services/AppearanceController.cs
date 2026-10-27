using Microsoft.UI.Windowing;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;
using Windows.UI;
using Windows.UI.ViewManagement;

namespace Sunno.Services;

/// <summary>
/// Themes the existing window, without touching capture or the speech engine.
/// System remains an inherited XAML theme, so Windows changes apply while running.
/// </summary>
public sealed class AppearanceController : IDisposable
{
    private readonly Window _window;
    private readonly FrameworkElement _root;
    private readonly DispatcherQueue _dispatcher;
    private readonly UISettings _uiSettings = new();
    private readonly AccessibilitySettings _accessibility = new();
    private string _preference = AppearancePreference.System;
    private volatile bool _disposed;

    public ElementTheme RequestedTheme => _preference switch
    {
        AppearancePreference.Light => ElementTheme.Light,
        AppearancePreference.Dark => ElementTheme.Dark,
        _ => ElementTheme.Default,
    };

    public AppearanceController(Window window, FrameworkElement root, string? preference)
    {
        _window = window;
        _root = root;
        _dispatcher = root.DispatcherQueue;
        _root.ActualThemeChanged += OnActualThemeChanged;
        _root.Loaded += OnLoaded;
        _window.Closed += OnClosed;
        // This desktop-compatible signal also covers contrast-color changes.
        // HighContrastChanged failed with 0x80070490 in our unpackaged UI check.
        _uiSettings.ColorValuesChanged += OnColorsChanged;
        Apply(preference);
    }

    public void Apply(string? preference)
    {
        if (_disposed) return;
        _preference = AppearancePreference.Normalize(preference);
        _root.RequestedTheme = RequestedTheme;
        RefreshChromeAndStatusColors();
    }

    private void OnActualThemeChanged(FrameworkElement sender, object args) => RefreshChromeAndStatusColors();
    private void OnLoaded(object sender, RoutedEventArgs args) => RefreshChromeAndStatusColors();
    private void OnClosed(object sender, WindowEventArgs args) => Dispose();
    private void OnColorsChanged(UISettings sender, object args) => QueueRefresh();

    private void QueueRefresh()
    {
        if (_disposed) return;
        _dispatcher.TryEnqueue(() =>
        {
            if (!_disposed) RefreshChromeAndStatusColors();
        });
    }

    private void RefreshChromeAndStatusColors()
    {
        if (_disposed) return;
        var highContrast = _accessibility.HighContrast;
        _window.AppWindow.TitleBar.PreferredTheme = highContrast ? TitleBarTheme.UseDefaultAppMode
            : _preference switch
            {
                AppearancePreference.Light => TitleBarTheme.Light,
                AppearancePreference.Dark => TitleBarTheme.Dark,
                _ => TitleBarTheme.UseDefaultAppMode,
            };

        var palette = highContrast
            ? AppearancePalette.ForHighContrast(ToArgb(_uiSettings.UIElementColor(UIElementType.WindowText)))
            : AppearancePalette.ForTheme(_root.ActualTheme == ElementTheme.Dark);

        // Keep the same brush objects: existing caption bindings then repaint too,
        // without rebuilding the transcript or losing its selection/scroll position.
        SetStatusColor("ClarityGood", palette.Good);
        SetStatusColor("ClarityMid", palette.Mid);
        SetStatusColor("ClarityLow", palette.Low);
        SetStatusColor("CaptionUncertain", highContrast ? palette.Good
            : AppearancePalette.UncertainText(_root.ActualTheme == ElementTheme.Dark));
    }

    private static uint ToArgb(Color color) =>
        (uint)color.A << 24 | (uint)color.R << 16 | (uint)color.G << 8 | color.B;

    private static void SetStatusColor(string key, uint argb)
    {
        if (Application.Current.Resources[key] is SolidColorBrush brush)
            brush.Color = Color.FromArgb((byte)(argb >> 24), (byte)(argb >> 16),
                                         (byte)(argb >> 8), (byte)argb);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _root.ActualThemeChanged -= OnActualThemeChanged;
        _root.Loaded -= OnLoaded;
        _window.Closed -= OnClosed;
        _uiSettings.ColorValuesChanged -= OnColorsChanged;
    }
}
