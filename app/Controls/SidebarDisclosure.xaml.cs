using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Animation;
using Sunno.Services;
using Windows.Foundation;

namespace Sunno.Controls;

public sealed partial class SidebarDisclosure : Expander
{
    private ToggleButton? _header;
    private Border? _clip;
    private RotateTransform? _chevron;
    private Storyboard? _transition;

    public SidebarDisclosure()
    {
        InitializeComponent();
        Expanding += (_, _) => Reveal(animate: true);
        Collapsed += (_, _) => Reveal(animate: true);
    }

    internal double HeaderHeight => _header?.ActualHeight ?? 0;

    protected override void OnApplyTemplate()
    {
        if (_clip is not null) _clip.GettingFocus -= OnBodyGettingFocus;
        _transition?.Stop();
        _transition = null;
        base.OnApplyTemplate();
        _header = (ToggleButton)GetTemplateChild("ExpanderHeader");
        _clip = (Border)GetTemplateChild("ExpanderContentClip");
        _chevron = (RotateTransform)((FontIcon)GetTemplateChild("Chevron")).RenderTransform;
        _clip.GettingFocus += OnBodyGettingFocus;
        Reveal(animate: false);
    }

    private void OnBodyGettingFocus(UIElement sender, GettingFocusEventArgs args)
    {
        if (!IsExpanded && _header is not null) args.TrySetNewFocusedElement(_header);
    }

    private void Reveal(bool animate)
    {
        if (_clip is null || _chevron is null || Content is not FrameworkElement body) return;

        var fromHeight = _clip.ActualHeight;
        var fromAngle = _chevron.Angle;
        _transition?.Stop();
        _transition = null;
        _clip.IsHitTestVisible = IsExpanded;
        if (!IsExpanded && XamlRoot is not null)
        {
            for (var focused = FocusManager.GetFocusedElement(XamlRoot) as UIElement;
                 focused is not null; focused = VisualTreeHelper.GetParent(focused) as UIElement)
            {
                if (focused != _clip) continue;
                _header?.Focus(FocusState.Programmatic);
                break;
            }
        }
        if (!animate || !IsLoaded || !AppearanceController.AnimationsEnabled)
        {
            FinishReveal();
            return;
        }

        _clip.Visibility = Visibility.Visible;
        var targetHeight = 0.0;
        if (IsExpanded)
        {
            body.Measure(new Size(ActualWidth > 0 ? ActualWidth : double.PositiveInfinity, double.PositiveInfinity));
            targetHeight = body.DesiredSize.Height;
        }
        var story = new Storyboard();
        story.Children.Add(Animate(_clip, "Height", fromHeight, targetHeight));
        story.Children.Add(Animate(_chevron, "Angle", fromAngle, IsExpanded ? 180 : 0));
        story.Completed += (_, _) =>
        {
            if (_transition != story) return;
            _transition = null;
            story.Stop();
            FinishReveal();
        };
        _transition = story;
        story.Begin();
    }

    private void FinishReveal()
    {
        _chevron!.Angle = IsExpanded ? 180 : 0;
        // Auto lets an open list follow new models, wrapped text, and window resizing.
        _clip!.Height = IsExpanded ? double.NaN : 0;
        _clip.Visibility = IsExpanded ? Visibility.Visible : Visibility.Collapsed;
    }

    private static DoubleAnimation Animate(DependencyObject target, string property, double from, double to)
    {
        var animation = new DoubleAnimation
        {
            From = from, To = to, Duration = new Duration(TimeSpan.FromMilliseconds(220)),
            EnableDependentAnimation = true,
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut }
        };
        Storyboard.SetTarget(animation, target);
        Storyboard.SetTargetProperty(animation, property);
        return animation;
    }
}
