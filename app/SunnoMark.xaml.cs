using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media.Animation;
using Windows.UI.ViewManagement;

namespace Sunno;

/// <summary>
/// The Sunno mark, drawn as vectors so it stays crisp at any scale and can take the theme's
/// ink colour — a bitmap would need a copy per size and would look wrong in high contrast.
///
/// Doubles as the app's loading indicator: with <see cref="IsAnimating"/> set, the caption
/// lines write themselves in and clear again. That is deliberately not a "listening"
/// animation — while the engine loads there is no microphone open, so the artwork shows
/// captions being composed, not sound arriving.
/// </summary>
public sealed partial class SunnoMark : UserControl
{
    private static readonly UISettings Settings = new();

    private readonly Storyboard _typing;
    private bool _isAnimating;

    public SunnoMark()
    {
        InitializeComponent();
        _typing = (Storyboard)Resources["Typing"];
        Loaded += (_, _) => Apply();
    }

    /// <summary>
    /// Whether the mark is writing itself. Stopping restores the XAML values, which draw the
    /// mark complete — so a paused or finished state is a finished caption block, not a
    /// half-written one frozen wherever the animation happened to be.
    /// </summary>
    public bool IsAnimating
    {
        get => _isAnimating;
        set
        {
            if (_isAnimating == value) return;
            _isAnimating = value;
            Apply();
        }
    }

    private void Apply()
    {
        // Begin() before the tree is loaded silently does nothing and leaves the mark blank,
        // so the property only records intent until Loaded has run.
        if (!IsLoaded) return;
        if (_isAnimating && AnimationsAllowed()) _typing.Begin();
        else _typing.Stop();
    }

    /// <summary>
    /// Honour "Play animations in Windows". A looping animation is exactly what that setting
    /// exists to suppress, and the loop ends on a cleared bubble — so a caller that ignored
    /// the setting and had the animation skipped would be left showing an empty mark.
    /// </summary>
    private static bool AnimationsAllowed()
    {
        try { return Settings.AnimationsEnabled; }
        catch { return true; }   // no reason to lose the animation over a failed query
    }
}
