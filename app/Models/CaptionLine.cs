using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Sunno.Models;

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
    private IReadOnlyList<CaptionWord> _words = Array.Empty<CaptionWord>();
    private DateTimeOffset? _spokenAt;

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
                Notify(nameof(ShowMeta));
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
                // DisplayLabel drives HasSpeaker, which drives ShowMeta. A line that becomes
                // "You" gains a label it did not have.
                Notify(nameof(HasSpeaker));
                Notify(nameof(ShowMeta));
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

    /// <summary>
    /// The user's preference, shared by every line rather than copied into each one: a line
    /// created after the toggle was flipped must agree with the lines already on screen, and a
    /// per-instance copy could only be set at construction.
    /// </summary>
    public static bool ClarityEnabled { get; set; } = true;

    public bool ShowClarity => ClarityEnabled && _isSelf && _isFinal && _clarity is not null;

    /// <summary>Re-evaluate after the shared preference changes; a static has no notification
    /// of its own, so the window tells each line to look again.</summary>
    public void RefreshClarity() => Notify(nameof(ShowClarity));

    /// <summary>
    /// Compact mode is on, so lines carry no speaker, time or clarity.
    ///
    /// Shared across lines for the same reason ClarityEnabled is: a line that arrives after
    /// the toggle has to agree with the ones already on screen.
    /// </summary>
    public static bool CompactMode { get; set; }

    /// <summary>
    /// Whether the row above the words is drawn at all.
    ///
    /// In a strip a few hundred pixels wide, the speaker, the timestamp and the clarity badge
    /// take more room than the sentence they describe. Compact drops all three, which is what
    /// "no labels, no frills" means, and the words are what someone is reading anyway.
    ///
    /// Display only. ToPlainText still keys on HasSpeaker, so copying out of a compact window
    /// yields the same "[time] Speaker: text" as copying out of a full one. Someone pasting a
    /// conversation into a note wants to know who said what, whatever size the window was.
    /// </summary>
    public bool ShowMeta => HasSpeaker && !CompactMode;

    /// <summary>Companion to RefreshClarity, for the same reason.</summary>
    public void RefreshMeta() => Notify(nameof(ShowMeta));

    /// <summary>
    /// Per-word confidence, used to mark uncertain words. Empty on provisional lines, which
    /// are replaced within a second anyway and aren't worth the extra decode work.
    /// </summary>
    public IReadOnlyList<CaptionWord> Words
    {
        get => _words;
        set => Set(ref _words, value);
    }

    /// <summary>When the utterance was spoken, not when it finished decoding.</summary>
    public DateTimeOffset? SpokenAt
    {
        get => _spokenAt;
        set
        {
            if (Set(ref _spokenAt, value)) Notify(nameof(TimeLabel));
        }
    }

    public string TimeLabel => _spokenAt?.ToLocalTime().ToString("HH:mm") ?? string.Empty;

    /// <summary>Plain text for the clipboard, including who said it and when.</summary>
    public string ToPlainText() =>
        HasSpeaker ? $"[{TimeLabel}] {DisplayLabel}: {Text}" : Text;

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

/// <summary>
/// One decoded word and how sure the model was of it.
///
/// <paramref name="Probability"/> is faster-whisper's per-word probability. Measured on clean
/// speech these sit at 0.97-1.00, while genuinely ambiguous words drop sharply, so the
/// threshold has a wide gap to sit in.
/// </summary>
public sealed record CaptionWord(string Text, double Probability)
{
    public const double UncertainBelow = 0.55;

    public bool IsUncertain => Probability < UncertainBelow;

    /// <summary>Percentage, for display on hover.</summary>
    public double Confidence => Probability * 100.0;
}
