using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Sunno.Controls;

public sealed partial class SkipToCurrentButton : UserControl
{
    public event RoutedEventHandler? Click;
    public SkipToCurrentButton() => InitializeComponent();
    private void OnClick(object sender, RoutedEventArgs e) => Click?.Invoke(this, e);
}
