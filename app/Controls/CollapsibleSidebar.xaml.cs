using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Markup;

namespace Sunno.Controls;

[ContentProperty(Name = nameof(PaneContent))]
public sealed partial class CollapsibleSidebar : UserControl
{
    public static readonly DependencyProperty PaneContentProperty = DependencyProperty.Register(
        nameof(PaneContent), typeof(UIElement), typeof(CollapsibleSidebar), new PropertyMetadata(null));

    public static readonly DependencyProperty IsCollapsedProperty = DependencyProperty.Register(
        nameof(IsCollapsed), typeof(bool), typeof(CollapsibleSidebar),
        new PropertyMetadata(false, (sender, _) => ((CollapsibleSidebar)sender).UpdateState()));

    public UIElement? PaneContent
    {
        get => (UIElement?)GetValue(PaneContentProperty);
        set => SetValue(PaneContentProperty, value);
    }

    public bool IsCollapsed
    {
        get => (bool)GetValue(IsCollapsedProperty);
        set => SetValue(IsCollapsedProperty, value);
    }

    public event EventHandler? CollapsedChanged;

    public CollapsibleSidebar()
    {
        InitializeComponent();
        UpdateState();
    }

    private void OnToggle(object sender, RoutedEventArgs e) => IsCollapsed = !IsCollapsed;

    private void UpdateState()
    {
        if (SidebarToggle is null) return;
        // Keep content mounted so folding does not reset the model picker or speaker list.
        Width = IsCollapsed ? 40 : 240;
        PanePresenter.Visibility = Heading.Visibility = IsCollapsed
            ? Visibility.Collapsed : Visibility.Visible;
        Chevron.Glyph = IsCollapsed ? "\uE76C" : "\uE76B";
        var label = IsCollapsed ? "Expand sidebar" : "Collapse sidebar";
        AutomationProperties.SetName(SidebarToggle, label);
        ToolTipService.SetToolTip(SidebarToggle, label);
        CollapsedChanged?.Invoke(this, EventArgs.Empty);
    }
}
