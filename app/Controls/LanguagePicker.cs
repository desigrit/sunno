using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Windows.System;

namespace Sunno.Controls;

public sealed class LanguagePicker : Button
{
    private sealed record Option(string Code, string Name, string Native)
    {
        public override string ToString() => Name;
    }

    private readonly TextBlock _title = new() { VerticalAlignment = VerticalAlignment.Center };
    private readonly TextBlock _value = new() { TextTrimming = TextTrimming.CharacterEllipsis,
        VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Right };
    private readonly FontIcon _chevron = new() { Glyph = "\uE70D", FontSize = 10 };
    private readonly TextBox _search = new() { PlaceholderText = "Search languages" };
    private readonly ListView _list = new() { MaxHeight = 250, IsItemClickEnabled = true };
    private readonly TextBlock _empty = new() { Text = "No matching languages", Margin = new Thickness(6) };
    private readonly TextBlock _download = new() { Text = "First use downloads a 638 MB translation model.",
        TextWrapping = TextWrapping.Wrap, FontSize = 12, Margin = new Thickness(6, 4, 6, 0) };
    private readonly Flyout _menu = new();
    private Option[] _options = Array.Empty<Option>();
    private string _selected = "";
    public event Action<string>? Chosen;

    public LanguagePicker()
    {
        HorizontalAlignment = HorizontalAlignment.Stretch;
        HorizontalContentAlignment = HorizontalAlignment.Stretch;
        Margin = new Thickness(8, 0, 8, 0);
        Padding = new Thickness(8, 4, 8, 4);
        Style = (Style)Application.Current.Resources["SubtleButtonStyle"];
        var row = new Grid { ColumnSpacing = 6 };
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        row.Children.Add(_title);
        Grid.SetColumn(_value, 1);
        row.Children.Add(_value);
        Grid.SetColumn(_chevron, 2);
        row.Children.Add(_chevron);
        Content = row;
        var panel = new StackPanel { Width = 228, Spacing = 6 };
        panel.Children.Add(_search);
        panel.Children.Add(_list);
        panel.Children.Add(_empty);
        panel.Children.Add(_download);
        _menu.Content = panel;
        Flyout = _menu;
        _menu.Opened += (_, _) => { _search.Text = ""; Filter(); _search.Focus(FocusState.Programmatic); };
        _search.TextChanged += (_, _) => Filter();
        _search.KeyDown += OnSearchKey;
        _list.ItemClick += (_, e) => Select((Option)e.ClickedItem);
        AutomationProperties.SetName(_search, "Search languages");
    }

    public void Configure(string title, IEnumerable<string> codes, string selected, string? detected = null,
                          bool preparing = false)
    {
        _selected = selected;
        _download.Visibility = title == "Translate to" && selected == "off" ? Visibility.Visible : Visibility.Collapsed;
        _title.Text = title;
        _options = codes.Select(code => new Option(code, LanguageName(code), LanguageName(code, native: true)))
            .OrderBy(o => o.Code is "off" or "auto" ? 0 : 1)
            .ThenBy(o => o.Name, StringComparer.CurrentCultureIgnoreCase).ToArray();
        var value = selected == "auto" && detected is not null ? $"Auto ({LanguageName(detected)})" : LanguageName(selected);
        _value.Text = preparing ? $"{value}…" : value;
        _chevron.Visibility = _options.Length > 1 ? Visibility.Visible : Visibility.Collapsed;
        IsEnabled = _options.Length > 1;
        AutomationProperties.SetName(this, $"{title}, {_value.Text}");
        AutomationProperties.SetName(_list, title);
        ToolTipService.SetToolTip(this, title == "Translate to" && selected == "off"
            ? "Translate completed captions locally. The first language downloads a 638 MB model." : $"{title}, {value}");
        if (_menu.IsOpen) Filter();
    }

    private static string LanguageName(string code, bool native = false)
    {
        if (code is "auto" or "off") return code == "auto" ? "Auto" : "Off";
        try
        {
            var culture = CultureInfo.GetCultureInfo(code == "jw" ? "jv" : code);
            return native ? culture.NativeName : culture.DisplayName;
        }
        catch (CultureNotFoundException) { return code; }
    }

    private void Filter()
    {
        string query = _search.Text.Trim();
        var options = _options.Where(o => new[] { o.Code, o.Name, o.Native }.Any(s =>
            CultureInfo.CurrentCulture.CompareInfo.IndexOf(s, query,
                CompareOptions.IgnoreCase | CompareOptions.IgnoreNonSpace) >= 0)).ToArray();
        _list.ItemsSource = options;
        _list.SelectedItem = options.FirstOrDefault(o => o.Code == _selected);
        _empty.Visibility = options.Length == 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    private void OnSearchKey(object sender, KeyRoutedEventArgs e)
    {
        if (_list.Items.FirstOrDefault() is not Option first) return;
        if (e.Key == VirtualKey.Enter) { Select(first); e.Handled = true; }
        if (e.Key == VirtualKey.Down)
        {
            _list.SelectedIndex = 0;
            _list.Focus(FocusState.Keyboard);
            e.Handled = true;
        }
    }

    private void Select(Option option)
    {
        _menu.Hide();
        Chosen?.Invoke(option.Code);
    }
}

