using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Sunno.Controls;

public sealed partial class AppUpdatePanel : UserControl
{
    public event RoutedEventHandler? ActionRequested;
    public event RoutedEventHandler? CancelRequested;
    public event RoutedEventHandler? CheckRequested;
    public event RoutedEventHandler? ReleasesRequested;
    public AppUpdatePanel() => InitializeComponent();
    private void OnAction(object sender, RoutedEventArgs e) => ActionRequested?.Invoke(this, e);
    private void OnCancel(object sender, RoutedEventArgs e) => CancelRequested?.Invoke(this, e);
    private void OnCheckAgain(object sender, RoutedEventArgs e) => CheckRequested?.Invoke(this, e);
    private void OnViewReleases(object sender, RoutedEventArgs e) => ReleasesRequested?.Invoke(this, e);

    public void Show(string action, string message = "", bool busy = false, bool enabled = true, double? progress = null)
    {
        UpdateAction.Content = action;
        UpdateAction.IsEnabled = enabled && !busy;
        UpdateOptions.IsEnabled = !busy;
        CancelAction.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
        DownloadProgress.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
        DownloadProgress.IsIndeterminate = progress is null;
        if (progress is double value) DownloadProgress.Value = Math.Clamp(value, 0, 100);
        UpdateStatus.Text = message;
        UpdateStatus.Visibility = string.IsNullOrEmpty(message) ? Visibility.Collapsed : Visibility.Visible;
    }
}
