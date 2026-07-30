using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace LiveCaptions.Models;

/// <summary>A row in the Speakers pane. Observable so renames update in place.</summary>
public sealed class SpeakerRow : INotifyPropertyChanged
{
    private string _label = string.Empty;
    private bool _isSelf;

    public int Id { get; set; }

    public string Label
    {
        get => _label;
        set => Set(ref _label, value);
    }

    public bool IsSelf
    {
        get => _isSelf;
        set => Set(ref _isSelf, value);
    }

    public bool Named { get; set; }

    public int ColourIndex => Id % 8;

    public event PropertyChangedEventHandler? PropertyChanged;

    private void Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return;
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
