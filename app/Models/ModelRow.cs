using System.ComponentModel;
using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;

namespace Sunno.Models;

/// <summary>
/// A selectable speech model in the left pane.
///
/// The row is two columns: name and description on the left, and — only when the model isn't
/// on disk — its download size and a download glyph on the right. Keeping the size out of the
/// description line means the description is always readable, and an un-downloaded model is
/// recognisable at a glance without reading anything.
/// </summary>
public sealed class ModelRow : INotifyPropertyChanged
{
    private bool _isSelected;
    private bool _isBusy;
    private bool _isIndeterminate;
    private double _progress;
    private bool _available;
    private bool _inUse;

    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Detail { get; set; } = string.Empty;
    public int ApproxMb { get; set; }

    /// <summary>
    /// Expected delay between someone finishing a sentence and its caption appearing, e.g.
    /// "about 0.7s behind". Measured per model per device, because the spread is what
    /// decides whether a choice is usable: the same model runs about 0.6s behind on a GPU
    /// and about 4.5s behind on CPU, and 4.5s is fine for captioning a recorded video but
    /// useless for following a conversation.
    /// </summary>
    public string LagText { get; set; } = string.Empty;

    /// <summary>Whether that delay is short enough to follow live conversation.</summary>
    public bool Responsive { get; set; } = true;

    /// <summary>Already downloaded.</summary>
    public bool Available
    {
        get => _available;
        set
        {
            if (!Set(ref _available, value)) return;
            Notify(nameof(Tooltip));
            Notify(nameof(DownloadHintVisibility));
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
    /// which would leave a stale radio contradicting the model that actually loaded.
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
            Notify(nameof(SecondaryVisibility));
            Notify(nameof(DownloadHintVisibility));
            Notify(nameof(IsEnabled));
        }
    }

    /// <summary>Currently loaded in the engine.</summary>
    public bool InUse
    {
        get => _inUse;
        set
        {
            if (Set(ref _inUse, value)) Notify(nameof(SecondaryText));
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

    /// <summary>The description, or "In use" for the loaded model.</summary>
    public string SecondaryText => _inUse ? "In use" : Detail;

    /// <summary>
    /// The speed line under the description. Left blank rather than showing a placeholder
    /// when the backend didn't report one, so an older backend degrades to the previous
    /// layout instead of to an empty row that looks broken.
    /// </summary>
    public Visibility LagVisibility =>
        !_isBusy && !string.IsNullOrEmpty(LagText) ? Visibility.Visible : Visibility.Collapsed;

    /// <summary>e.g. "1.5 GB" — shown on the right only when a download is needed.</summary>
    public string SizeLabel => FormatSize(ApproxMb);

    public Visibility BusyVisibility => _isBusy ? Visibility.Visible : Visibility.Collapsed;
    public Visibility SecondaryVisibility => _isBusy ? Visibility.Collapsed : Visibility.Visible;

    /// <summary>Size and download glyph: only when it isn't here and isn't already coming.</summary>
    public Visibility DownloadHintVisibility =>
        !_available && !_isBusy ? Visibility.Visible : Visibility.Collapsed;

    /// <summary>A model mid-download can't be chosen again without confusing the backend.</summary>
    public bool IsEnabled => !_isBusy;

    /// <summary>Full text for hover, since the pane is 240px and lines trim.</summary>
    public string Tooltip
    {
        get
        {
            var text = string.IsNullOrEmpty(Detail) ? Name : $"{Name}\n{Detail}";
            if (!string.IsNullOrEmpty(LagText))
            {
                text += $"\nCaptions appear {LagText}";
                if (!Responsive)
                    text += " — fine for video, too slow to follow a conversation";
            }
            return _available ? text : $"{text}\n{SizeLabel} download";
        }
    }

    /// <summary>Return to the resting state after a download finishes or is abandoned.</summary>
    public void Refresh()
    {
        IsBusy = false;
        IsIndeterminate = false;
        Progress = 0;
        Notify(nameof(SecondaryText));
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
