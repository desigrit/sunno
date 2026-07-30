using System.ComponentModel;
using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;

namespace Sunno.Models;

/// <summary>
/// A selectable speech model in the left pane.
///
/// The secondary line does triple duty depending on state: the download size when the model
/// isn't on disk, live progress while it downloads, and how it compares once it's resident.
/// Keeping that in one property avoids three overlapping elements fighting for the same 240px.
/// </summary>
public sealed class ModelRow : INotifyPropertyChanged
{
    private bool _isSelected;
    private bool _isBusy;
    private bool _available;
    private string _status = string.Empty;

    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Detail { get; set; } = string.Empty;
    public int ApproxMb { get; set; }

    /// <summary>Already downloaded.</summary>
    public bool Available
    {
        get => _available;
        set { if (Set(ref _available, value)) Refresh(); }
    }

    public bool IsSelected
    {
        get => _isSelected;
        set => Set(ref _isSelected, value);
    }

    /// <summary>Downloading, or waiting for the engine to reload onto this model.</summary>
    public bool IsBusy
    {
        get => _isBusy;
        set
        {
            if (!Set(ref _isBusy, value)) return;
            Notify(nameof(BusyVisibility));
            Notify(nameof(IsEnabled));
        }
    }

    public string Status
    {
        get => _status;
        set => Set(ref _status, value);
    }

    public Visibility BusyVisibility => _isBusy ? Visibility.Visible : Visibility.Collapsed;

    /// <summary>A model mid-download can't be chosen again without confusing the backend.</summary>
    public bool IsEnabled => !_isBusy;

    /// <summary>Restore the resting caption after a download finishes or is abandoned.</summary>
    public void Refresh() => Status = _available ? Detail : $"{FormatSize(ApproxMb)} download";

    public void ShowProgress(double percent) => Status = $"Downloading… {percent:0}%";

    private static string FormatSize(int mb) =>
        mb >= 1024 ? $"{mb / 1024.0:0.#} GB" : $"{mb} MB";

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
