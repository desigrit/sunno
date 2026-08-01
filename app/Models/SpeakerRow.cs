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
        set
        {
            if (!Set(ref _label, value)) return;
            Notify(nameof(Tooltip));
            // The Settings page repeats an Edit and a Delete button per row, so each needs a
            // name of its own: a screen reader otherwise reads out a column of buttons all
            // called "Delete" with no way to tell whose is whose.
            Notify(nameof(EditActionName));
            Notify(nameof(DeleteActionName));
        }
    }

    public string EditActionName => $"Edit {DisplayName}";
    public string DeleteActionName => $"Delete {DisplayName}";

    private string DisplayName => string.IsNullOrWhiteSpace(_label) ? "this speaker" : _label;

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
