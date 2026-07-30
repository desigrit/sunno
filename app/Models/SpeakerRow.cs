using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Sunno.Models;

/// <summary>A row in the Speakers pane. Observable so renames update in place.</summary>
public sealed class SpeakerRow : INotifyPropertyChanged
{
    private string _label = string.Empty;
    private bool _isSelf;

    public int Id { get; set; }

    public string Label
    {
        get => _label;
        set { if (Set(ref _label, value)) Notify(nameof(Tooltip)); }
    }

    /// <summary>Full name plus the affordance, since long names trim in a 240px pane.</summary>
    public string Tooltip => string.IsNullOrWhiteSpace(_label)
        ? "Click to rename or mark as you"
        : $"{_label}\nClick to rename or mark as you";

    public bool IsSelf
    {
        get => _isSelf;
        set => Set(ref _isSelf, value);
    }

    public bool Named { get; set; }

    public int ColourIndex => Id % 8;

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return false;
        field = value;
        Notify(name);
        return true;
    }

    private void Notify(string? name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
