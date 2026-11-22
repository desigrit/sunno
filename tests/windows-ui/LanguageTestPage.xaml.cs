using Microsoft.UI.Xaml.Controls;

namespace Sunno;

public sealed partial class LanguageTestPage : Page
{
    internal Grid TestSurface => Surface;

    public LanguageTestPage() => InitializeComponent();
}
