using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Sunno.Controls;

namespace Sunno.Services;

/// <summary>
/// Following is reader intent, not a property inferred from layout-driven movement.
/// Only navigation can pause it; revisions, finalization and history trimming cannot.
/// </summary>
public sealed class CaptionScrollFollower : IDisposable
{
    public const double NearBottom = 80;
    private readonly ScrollViewer _viewer;
    private readonly CaptionScrollViewer _host;
    private readonly FrameworkElement _content;
    private readonly PointerEventHandler _wheel;
    private readonly KeyEventHandler _key;
    private bool _queued, _changingView, _disposed, _manipulating;
    private bool _navigationPending;
    private int _direction;
    private double _lastOffset;

    public bool IsFollowing { get; private set; } = true;
    public bool CanSkipToCurrent => !IsFollowing && _viewer.ScrollableHeight - _viewer.VerticalOffset > .5;
    public event Action? Changed;

    public CaptionScrollFollower(CaptionScrollViewer host, FrameworkElement content)
    {
        _host = host;
        var viewer = host.Viewer;
        _viewer = viewer;
        _content = content;
        _lastOffset = viewer.VerticalOffset;
        host.ScrollRequested += OnScrollRequested;
        host.ScrollCompleted += OnScrollCompleted;
        viewer.ViewChanged += OnViewChanged;
        viewer.DirectManipulationStarted += OnManipulationStarted;
        viewer.DirectManipulationCompleted += OnManipulationCompleted;
        viewer.SizeChanged += OnSizeChanged;
        content.SizeChanged += OnSizeChanged;
        viewer.Loaded += OnLoaded;
        _wheel = (_, e) =>
        {
            var properties = e.GetCurrentPoint(viewer).Properties;
            if (!properties.IsHorizontalMouseWheel && properties.MouseWheelDelta != 0)
                OnScrollRequested(properties.MouseWheelDelta > 0 ? -1 : 1, null);
        };
        _key = (_, e) =>
        {
            if (e.Key is Windows.System.VirtualKey.Up or Windows.System.VirtualKey.PageUp or Windows.System.VirtualKey.Home)
                OnScrollRequested(-1, null);
            else if (e.Key == Windows.System.VirtualKey.End)
                OnScrollRequested(1, viewer.ScrollableHeight);
            else if (e.Key is Windows.System.VirtualKey.Down or Windows.System.VirtualKey.PageDown)
                OnScrollRequested(1, null);
        };
        viewer.AddHandler(UIElement.PointerWheelChangedEvent, _wheel, true);
        viewer.AddHandler(UIElement.KeyDownEvent, _key, true);
    }

    private bool IsNearBottom(double offset) => _viewer.ScrollableHeight - offset <= NearBottom;

    private void SetFollowing(bool value)
    {
        IsFollowing = value;
        Changed?.Invoke();
    }

    private void OnScrollRequested(int direction, double? target)
    {
        if (_disposed || _viewer.ScrollableHeight <= 0) return;
        _direction = direction;
        _navigationPending = true;
        if (direction < 0) SetFollowing(false);
        else if (target is double offset)
            SetFollowing(IsNearBottom(offset));
        // A content reflow has no navigation signal and cannot overwrite this preference.
    }

    private void OnViewChanged(object? sender, ScrollViewerViewChangedEventArgs e)
    {
        if (_disposed) return;
        var offset = _viewer.VerticalOffset;
        if (!_changingView)
        {
            if (_manipulating && Math.Abs(offset - _lastOffset) > .5)
            {
                _direction = Math.Sign(offset - _lastOffset);
                SetFollowing(_direction > 0 && IsNearBottom(offset));
            }
            else if (_navigationPending && _direction > 0)
                SetFollowing(IsNearBottom(offset));
        }
        _lastOffset = offset;
        if (!e.IsIntermediate) _navigationPending = false;
        Changed?.Invoke();
        // ChangeView and layout finish asynchronously. Follow through later anchor
        // adjustments as well as the initial text update, until the reader scrolls away.
        if (IsFollowing && !e.IsIntermediate && _viewer.ScrollableHeight - offset > .5)
            RequestFollow();
    }

    private void OnManipulationStarted(object? sender, object e)
    {
        _manipulating = true;
        _lastOffset = _viewer.VerticalOffset;
    }

    private void OnManipulationCompleted(object? sender, object e)
    {
        _manipulating = false;
        OnScrollCompleted();
    }

    private void OnScrollCompleted()
    {
        _navigationPending = false;
        Changed?.Invoke();
        RequestFollow();
    }

    private void OnLoaded(object sender, RoutedEventArgs e) => RequestFollow();
    private void OnSizeChanged(object sender, SizeChangedEventArgs e) => RequestFollow();

    public void RequestFollow()
    {
        if (_disposed || _queued || _changingView) return;
        _queued = _viewer.DispatcherQueue.TryEnqueue(DispatcherQueuePriority.Low, () =>
        {
            _queued = false;
            FollowAfterLayout();
        });
    }

    private void FollowAfterLayout()
    {
        if (_disposed || _changingView || _manipulating || _host.IsScrollBarDragging) return;
        _changingView = true;
        try
        {
            _viewer.UpdateLayout();
            if (IsFollowing && Math.Abs(_viewer.VerticalOffset - _viewer.ScrollableHeight) > .5)
                _viewer.ChangeView(null, _viewer.ScrollableHeight, null, disableAnimation: true);
            _lastOffset = _viewer.VerticalOffset;
        }
        finally { _changingView = false; }
        Changed?.Invoke();
    }

    public void Resume()
    {
        if (_disposed) return;
        _navigationPending = false;
        _direction = 0;
        SetFollowing(true);
        FollowAfterLayout();
        RequestFollow();
    }

    public void Reset() => Resume();

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _host.ScrollRequested -= OnScrollRequested;
        _host.ScrollCompleted -= OnScrollCompleted;
        _viewer.ViewChanged -= OnViewChanged;
        _viewer.DirectManipulationStarted -= OnManipulationStarted;
        _viewer.DirectManipulationCompleted -= OnManipulationCompleted;
        _viewer.SizeChanged -= OnSizeChanged;
        _content.SizeChanged -= OnSizeChanged;
        _viewer.Loaded -= OnLoaded;
        _viewer.RemoveHandler(UIElement.PointerWheelChangedEvent, _wheel);
        _viewer.RemoveHandler(UIElement.KeyDownEvent, _key);
        Changed = null;
    }
}
