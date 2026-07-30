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
    private bool _isIndeterminate;
    private double _progress;
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
        set
        {
            if (!Set(ref _available, value)) return;
            Notify(nameof(Tooltip));
            Refresh();
        }
    }

    /// <summary>
    /// Full text for hover, because the pane is 240px and both the name and the description
    /// routinely trim. Includes the download size, which the caption line only shows while the
    /// model is absent and hides entirely once a download starts.
    /// </summary>
    public string Tooltip
    {
        get
        {
            var text = string.IsNullOrEmpty(Detail) ? Name : $"{Name}\n{Detail}";
            return _available ? text : $"{text}\n{FormatSize(ApproxMb)} download";
        }
    }

    public bool IsSelected
    {
        get => _isSelected;
        set => Set(ref _isSelected, value);
    }

    /// <summary>
    /// Push the selection into the view unconditionally.
    ///
    /// The IsChecked binding is one-way, so clicking a radio leaves the control checked while
    /// this model still reads false. Assigning false back is then a no-op and raises nothing,
    /// which would leave a stale radio contradicting the model that actually loaded. Re-raising
    /// the change regardless is what pulls the control back into line.
    /// </summary>
    public void SetSelected(bool value)
    {
        _isSelected = value;
        Notify(nameof(IsSelected));
    }

    /// <summary>Downloading, or waiting for the engine to reload onto this model.</summary>
    public bool IsBusy
    {
        get => _isBusy;
        set
        {
            if (!Set(ref _isBusy, value)) return;
            Notify(nameof(BusyVisibility));
            Notify(nameof(StatusVisibility));
            Notify(nameof(IsEnabled));
        }
    }

    /// <summary>A reload has no measurable progress; a download does.</summary>
    public bool IsIndeterminate
    {
        get => _isIndeterminate;
        set => Set(ref _isIndeterminate, value);
    }

    public double Progress
    {
        get => _progress;
        set => Set(ref _progress, value);
    }

    public string Status
    {
        get => _status;
        set => Set(ref _status, value);
    }

    /// <summary>The bar replaces the caption rather than crowding in beside it.</summary>
    public Visibility BusyVisibility => _isBusy ? Visibility.Visible : Visibility.Collapsed;
    public Visibility StatusVisibility => _isBusy ? Visibility.Collapsed : Visibility.Visible;

    /// <summary>A model mid-download can't be chosen again without confusing the backend.</summary>
    public bool IsEnabled => !_isBusy;

    /// <summary>Restore the resting caption after a download finishes or is abandoned.</summary>
    public void Refresh()
    {
        IsBusy = false;
        IsIndeterminate = false;
        Progress = 0;
        Status = _available ? Detail : $"{FormatSize(ApproxMb)} download";
    }

    /// <summary>Fetching bytes: a bar that fills, with no number to watch.</summary>
    public void ShowProgress(double percent)
    {
        IsIndeterminate = false;
        Progress = Math.Clamp(percent, 0, 100);
        IsBusy = true;
    }

    /// <summary>Loading into the engine: real work, but nothing meaningful to measure.</summary>
    public void ShowLoading()
    {
        IsIndeterminate = true;
        IsBusy = true;
    }

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
