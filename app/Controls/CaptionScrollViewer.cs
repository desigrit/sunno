using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Automation.Provider;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Markup;
using Microsoft.UI.Xaml.Media;

namespace Sunno.Controls;

/// <summary>Native scrolling with explicit navigation signals, including accessibility.</summary>
[ContentProperty(Name = nameof(ScrollContent))]
public sealed class CaptionScrollViewer : UserControl
{
    public ScrollViewer Viewer { get; } = new();
    public object? ScrollContent { get => Viewer.Content; set => Viewer.Content = value; }
    public Thickness ScrollPadding { get => Viewer.Padding; set => Viewer.Padding = value; }
    public ScrollBarVisibility VerticalScrollBarVisibility
    { get => Viewer.VerticalScrollBarVisibility; set => Viewer.VerticalScrollBarVisibility = value; }
    public ScrollBarVisibility HorizontalScrollBarVisibility
    { get => Viewer.HorizontalScrollBarVisibility; set => Viewer.HorizontalScrollBarVisibility = value; }
    public double VerticalAnchorRatio
    { get => Viewer.VerticalAnchorRatio; set => Viewer.VerticalAnchorRatio = value; }
    internal event Action<int, double?>? ScrollRequested;
    internal event Action? ScrollCompleted;
    internal bool IsScrollBarDragging { get; private set; }
    private ScrollBar? _bar;
    private double _barOffset;
    private readonly PointerEventHandler _pressed, _released;

    public CaptionScrollViewer()
    {
        Content = Viewer;
        AutomationProperties.SetAccessibilityView(Viewer, AccessibilityView.Raw);
        _pressed = (_, _) => { _barOffset = Viewer.VerticalOffset; IsScrollBarDragging = true; };
        _released = (_, _) => EndBarScroll();
        Viewer.Loaded += (_, _) => AttachScrollBar();
        Viewer.Unloaded += (_, _) => DetachScrollBar();
    }

    private void DetachScrollBar()
    {
        if (_bar is not null)
        {
            _bar.Scroll -= OnBarScroll;
            _bar.RemoveHandler(PointerPressedEvent, _pressed);
            _bar.RemoveHandler(PointerReleasedEvent, _released);
            _bar.RemoveHandler(PointerCaptureLostEvent, _released);
        }
        _bar = null;
        IsScrollBarDragging = false;
    }

    private void AttachScrollBar()
    {
        DetachScrollBar();
        _bar = FindVerticalBar(Viewer);
        if (_bar is null) return;
        _bar.Scroll += OnBarScroll;
        _bar.AddHandler(PointerPressedEvent, _pressed, true);
        _bar.AddHandler(PointerReleasedEvent, _released, true);
        _bar.AddHandler(PointerCaptureLostEvent, _released, true);
    }

    private static ScrollBar? FindVerticalBar(DependencyObject parent)
    {
        for (var i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
        {
            var child = VisualTreeHelper.GetChild(parent, i);
            if (child is ScrollBar bar && bar.Orientation == Orientation.Vertical) return bar;
            if (FindVerticalBar(child) is { } found) return found;
        }
        return null;
    }

    private void OnBarScroll(object sender, ScrollEventArgs e)
    {
        if (e.ScrollEventType == ScrollEventType.EndScroll) { EndBarScroll(); return; }
        var direction = e.ScrollEventType switch
        {
            ScrollEventType.First or ScrollEventType.SmallDecrement or ScrollEventType.LargeDecrement => -1,
            ScrollEventType.Last or ScrollEventType.SmallIncrement or ScrollEventType.LargeIncrement => 1,
            _ => Math.Sign(e.NewValue - (IsScrollBarDragging ? _barOffset : Viewer.VerticalOffset)),
        };
        _barOffset = e.NewValue;
        ScrollRequested?.Invoke(direction, e.NewValue);
    }

    private void EndBarScroll()
    {
        IsScrollBarDragging = false;
        ScrollCompleted?.Invoke();
    }

    protected override AutomationPeer OnCreateAutomationPeer() => new CaptionScrollPeer(this);

    // Accessibility scroll requests are navigation too. Forward the native pattern's
    // methods and properties so screen readers retain the standard ScrollViewer behavior.
    private sealed class CaptionScrollPeer(CaptionScrollViewer viewer) : FrameworkElementAutomationPeer(viewer)
    {
        protected override AutomationControlType GetAutomationControlTypeCore() => AutomationControlType.Pane;
        protected override object GetPatternCore(PatternInterface pattern)
        {
            var original = FrameworkElementAutomationPeer.CreatePeerForElement(viewer.Viewer).GetPattern(pattern);
            return pattern == PatternInterface.Scroll && original is IScrollProvider provider
                ? new ScrollProvider(viewer, provider) : original;
        }
    }

    private sealed class ScrollProvider(CaptionScrollViewer viewer, IScrollProvider native) : IScrollProvider
    {
        public bool HorizontallyScrollable => native.HorizontallyScrollable;
        public bool VerticallyScrollable => native.VerticallyScrollable;
        public double HorizontalScrollPercent => native.HorizontalScrollPercent;
        public double VerticalScrollPercent => native.VerticalScrollPercent;
        public double HorizontalViewSize => native.HorizontalViewSize;
        public double VerticalViewSize => native.VerticalViewSize;
        public void Scroll(ScrollAmount horizontal, ScrollAmount vertical)
        {
            if (vertical != ScrollAmount.NoAmount && native.VerticallyScrollable)
                viewer.ScrollRequested?.Invoke(vertical is ScrollAmount.LargeDecrement or ScrollAmount.SmallDecrement ? -1 : 1, null);
            native.Scroll(horizontal, vertical);
        }
        public void SetScrollPercent(double horizontal, double vertical)
        {
            if (vertical is >= 0 and <= 100 && native.VerticallyScrollable)
            {
                var target = viewer.Viewer.ScrollableHeight * vertical / 100;
                viewer.ScrollRequested?.Invoke(Math.Sign(target - viewer.Viewer.VerticalOffset), target);
            }
            native.SetScrollPercent(horizontal, vertical);
        }
    }
}
