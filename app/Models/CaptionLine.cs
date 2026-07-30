using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace LiveCaptions.Models;

/// <summary>
/// One line in the transcript. Mutable because a line starts out provisional and is upgraded
/// in place to final text when the utterance ends — the grey-then-commit behaviour that BBC
/// and DCMP guidance recommend for live subtitles.
/// </summary>
public sealed class CaptionLine : INotifyPropertyChanged
{
    private string _text = string.Empty;
    private bool _isFinal;
    private string? _speakerLabel;
    private int? _speakerId;
    private bool _isSelf;
    private int? _clarity;

    // Not init-only: the XAML type-info generator emits a plain setter assignment.
    public int UtteranceId { get; set; }

    public string Text
    {
        get => _text;
        set => Set(ref _text, value);
    }

    public bool IsFinal
    {
        get => _isFinal;
        set
        {
            if (Set(ref _isFinal, value))
            {
                Notify(nameof(IsProvisional));
                Notify(nameof(ShowClarity));
            }
        }
    }

    public bool IsProvisional => !_isFinal;

    public string? SpeakerLabel
    {
        get => _speakerLabel;
        set
        {
            if (Set(ref _speakerLabel, value))
            {
                Notify(nameof(DisplayLabel));
                Notify(nameof(HasSpeaker));
            }
        }
    }

    public int? SpeakerId
    {
        get => _speakerId;
        set => Set(ref _speakerId, value);
    }

    /// <summary>True when this is the user's own speech.</summary>
    public bool IsSelf
    {
        get => _isSelf;
        set
        {
            if (Set(ref _isSelf, value))
            {
                Notify(nameof(DisplayLabel));
                Notify(nameof(ShowClarity));
            }
        }
    }

    /// <summary>
    /// Decode confidence 0-100, shown only on the user's own lines as speech-clarity
    /// feedback. On other people's lines it would just be noise.
    /// </summary>
    public int? Clarity
    {
        get => _clarity;
        set
        {
            if (Set(ref _clarity, value)) Notify(nameof(ShowClarity));
        }
    }

    public bool HasSpeaker => !string.IsNullOrEmpty(DisplayLabel);
    public string DisplayLabel => _isSelf ? "You" : _speakerLabel ?? string.Empty;
    public bool ShowClarity => _isSelf && _isFinal && _clarity is not null;

    /// <summary>Palette index so each speaker keeps a stable colour.</summary>
    public int ColourIndex => (_speakerId ?? 0) % 8;

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

/// <summary>A speaker the backend has discovered, or that the user has named.</summary>
public sealed record SpeakerInfo(int Id, string Label, bool Named, bool IsSelf);
