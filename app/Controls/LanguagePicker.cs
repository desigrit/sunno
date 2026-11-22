using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Windows.Foundation;
using Windows.System;

namespace Sunno.Controls;

public sealed class LanguagePicker : Button
{
    private sealed record Option(string Code, string Name, string Native)
    {
        public override string ToString() => Name;
    }

    private readonly TextBlock _title = new() { TextWrapping = TextWrapping.Wrap, VerticalAlignment = VerticalAlignment.Center };
    private readonly TextBlock _value = new() { TextWrapping = TextWrapping.Wrap, VerticalAlignment = VerticalAlignment.Center };
    private readonly FontIcon _chevron = new() { Glyph = "\uE70D", FontSize = 10 };
    private readonly TextBox _search = new() { PlaceholderText = "Search languages" };
    private readonly ListView _list = new() { MaxHeight = 250, IsItemClickEnabled = true };
    private readonly TextBlock _empty = new() { Text = "No matching languages", Margin = new Thickness(6) };
    private readonly TextBlock _download = new() { Text = "First use downloads a 638 MB translation model.",
        TextWrapping = TextWrapping.Wrap, FontSize = 12, Margin = new Thickness(6, 4, 6, 0) };
    private readonly Flyout _menu = new();
    private Option[] _options = Array.Empty<Option>();
    private Option[] _matches = Array.Empty<Option>();
    private string _selected = "";
    public event Action<string>? Chosen;

    public LanguagePicker()
    {
        HorizontalAlignment = HorizontalAlignment.Stretch;
        HorizontalContentAlignment = HorizontalAlignment.Stretch;
        Margin = new Thickness(8, 0, 8, 0);
        Padding = new Thickness(8, 4, 8, 4);
        if (Application.Current.Resources.TryGetValue("SubtleButtonStyle", out var subtle) && subtle is Style style)
            Style = style;
        Content = new LanguageRow(_title, _value, _chevron);
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
        _list.ItemClick += (_, e) =>
        {
            if (_matches.FirstOrDefault(o => o.Name == e.ClickedItem as string) is Option chosen) Select(chosen);
        };
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

    // Keep short values on one line and let the full selection wrap below the label
    // when the sidebar is narrow or the user increases the Windows text size.
    private sealed class LanguageRow : Panel
    {
        private const double Gap = 6;
        private readonly TextBlock _label, _selection;
        private readonly FontIcon _arrow;
        private bool _stacked;

        public LanguageRow(TextBlock label, TextBlock selection, FontIcon arrow)
        {
            _label = label;
            _selection = selection;
            _arrow = arrow;
            Children.Add(label);
            Children.Add(selection);
            Children.Add(arrow);
        }

        private double ArrowSpace => _arrow.Visibility == Visibility.Visible ? _arrow.DesiredSize.Width + Gap : 0;

        protected override Size MeasureOverride(Size availableSize)
        {
            var unrestricted = new Size(double.PositiveInfinity, double.PositiveInfinity);
            _arrow.Measure(unrestricted);
            _label.Measure(unrestricted);
            _selection.Measure(unrestricted);
            var inline = _label.DesiredSize.Width + Gap + _selection.DesiredSize.Width;
            var textWidth = Math.Max(0, availableSize.Width - ArrowSpace);
            _stacked = inline > textWidth;
            if (_stacked)
            {
                var limit = new Size(textWidth, double.PositiveInfinity);
                _label.Measure(limit);
                _selection.Measure(limit);
            }
            var width = _stacked ? Math.Max(_label.DesiredSize.Width, _selection.DesiredSize.Width) : inline;
            var height = _stacked ? _label.DesiredSize.Height + _selection.DesiredSize.Height
                                  : Math.Max(_label.DesiredSize.Height, _selection.DesiredSize.Height);
            return new Size(Math.Min(availableSize.Width, width + ArrowSpace), Math.Max(height, _arrow.DesiredSize.Height));
        }

        protected override Size ArrangeOverride(Size finalSize)
        {
            var textWidth = Math.Max(0, finalSize.Width - ArrowSpace);
            if (_stacked)
            {
                _label.Arrange(new Rect(0, 0, textWidth, _label.DesiredSize.Height));
                _selection.Arrange(new Rect(0, _label.DesiredSize.Height, textWidth, _selection.DesiredSize.Height));
            }
            else
            {
                _label.Arrange(new Rect(0, 0, _label.DesiredSize.Width, finalSize.Height));
                var width = Math.Min(textWidth, _selection.DesiredSize.Width);
                _selection.Arrange(new Rect(textWidth - width, 0, width, finalSize.Height));
            }
            _arrow.Arrange(new Rect(Math.Max(0, finalSize.Width - _arrow.DesiredSize.Width), 0,
                                    _arrow.DesiredSize.Width, finalSize.Height));
            return finalSize;
        }
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
        var compare = CultureInfo.CurrentCulture.CompareInfo;
        const CompareOptions comparison = CompareOptions.IgnoreCase | CompareOptions.IgnoreNonSpace;
        int Rank(Option option)
        {
            int rank = 3;
            foreach (var value in new[] { option.Code, option.Name, option.Native })
            {
                if (compare.Compare(value, query, comparison) == 0) return 0;
                var index = compare.IndexOf(value, query, comparison);
                if (index >= 0) rank = Math.Min(rank, index == 0 ? 1 : 2);
            }
            return rank;
        }
        _matches = query.Length == 0 ? _options : _options.Select(o => (Option: o, Rank: Rank(o)))
            .Where(item => item.Rank < 3).OrderBy(item => item.Rank).Select(item => item.Option).ToArray();
        _list.ItemsSource = _matches.Select(o => o.Name).ToArray();
        _list.SelectedIndex = query.Length == 0 ? Array.FindIndex(_matches, o => o.Code == _selected)
            : _matches.Length == 0 ? -1 : 0;
        if (_list.SelectedItem != null) _list.ScrollIntoView(_list.SelectedItem);
        _empty.Visibility = _matches.Length == 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    private void OnSearchKey(object sender, KeyRoutedEventArgs e)
    {
        if (_matches.Length == 0) return;
        if (e.Key == VirtualKey.Enter && _list.SelectedIndex >= 0)
        {
            Select(_matches[_list.SelectedIndex]);
            e.Handled = true;
        }
        if (e.Key == VirtualKey.Down)
        {
            if (_list.SelectedIndex < 0) _list.SelectedIndex = 0;
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

