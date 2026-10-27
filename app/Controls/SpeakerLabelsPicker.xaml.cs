using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Sunno.Controls;

public sealed partial class SpeakerLabelsPicker : UserControl
{
    private bool _syncing;
    private bool _automatic;
    public event EventHandler<bool>? PreferenceChanged;

    public SpeakerLabelsPicker()
    {
        InitializeComponent();
        Automatic = false;
    }

    public bool Automatic
    {
        get => _automatic;
        set
        {
            _automatic = value;
            _syncing = true;
            try
            {
                AutomaticOption.IsChecked = value;
                OffOption.IsChecked = !value;
            }
            finally { _syncing = false; }
        }
    }

    private void OnOptionChecked(object sender, RoutedEventArgs args)
    {
        if (_syncing) return;
        var automatic = ReferenceEquals(sender, AutomaticOption);
        if (_automatic == automatic) return;
        _automatic = automatic;
        PreferenceChanged?.Invoke(this, automatic);
    }

    private void OnLayoutSizeChanged(object sender, SizeChangedEventArgs args)
    {
        var compact = args.NewSize.Width < 380;
        Grid.SetRow(OptionsContainer, compact ? 1 : 0);
        Grid.SetColumn(OptionsContainer, compact ? 0 : 1);
        Grid.SetColumnSpan(OptionsContainer, compact ? 2 : 1);
        OptionsContainer.HorizontalAlignment = compact ? HorizontalAlignment.Left : HorizontalAlignment.Right;
        OptionsContainer.Margin = compact ? new Thickness(0, 8, 0, 0) : new Thickness(0);
    }
}
