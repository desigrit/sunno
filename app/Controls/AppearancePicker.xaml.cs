using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Sunno.Services;

namespace Sunno.Controls;

public sealed partial class AppearancePicker : UserControl
{
    private bool _syncing;
    private string _preference = AppearancePreference.System;
    public event EventHandler<string>? PreferenceChanged;

    public AppearancePicker()
    {
        InitializeComponent();
        Preference = AppearancePreference.System;
    }

    public string Preference
    {
        get => _preference;
        set
        {
            _preference = AppearancePreference.Normalize(value);
            _syncing = true;
            try { ModePicker.SelectedIndex = AppearancePreference.IndexOf(_preference); }
            finally { _syncing = false; }
        }
    }

    private void OnSelectionChanged(object sender, SelectionChangedEventArgs args)
    {
        if (_syncing || ModePicker.SelectedItem is not RadioButton selected) return;
        var preference = AppearancePreference.Normalize(selected.Tag as string);
        // The repeater can complete a settings restore after its initial layout.
        if (_preference == preference) return;
        _preference = preference;
        PreferenceChanged?.Invoke(this, preference);
    }

    private void OnLayoutSizeChanged(object sender, SizeChangedEventArgs args)
    {
        // Use this card's width, not the whole window or display, for compact layouts.
        var compact = args.NewSize.Width < 420;
        Grid.SetRow(ModeContainer, compact ? 1 : 0);
        Grid.SetColumn(ModeContainer, compact ? 0 : 1);
        Grid.SetColumnSpan(ModeContainer, compact ? 2 : 1);
        ModeContainer.HorizontalAlignment = compact ? HorizontalAlignment.Left : HorizontalAlignment.Right;
        ModeContainer.Margin = compact ? new Thickness(0, 8, 0, 0) : new Thickness(0);
    }
}
