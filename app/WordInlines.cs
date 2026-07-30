using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Sunno.Models;

namespace Sunno;

/// <summary>
/// Renders a caption as individual word runs, marks uncertain ones, and reports each word's
/// confidence on hover.
///
/// Uses <see cref="RichTextBlock"/> rather than TextBlock for one reason: inlines are not
/// UIElements, so they receive no pointer input and cannot carry their own tooltip.
/// RichTextBlock exposes <c>GetPositionFromPoint</c>, which lets the pointer position be
/// mapped back to an exact word. Offsets are captured from each Run's own ContentStart at
/// build time rather than derived from character counts, so the mapping stays exact regardless
/// of how the text is composed.
/// </summary>
public static class WordInlines
{
    public static readonly DependencyProperty LineProperty =
        DependencyProperty.RegisterAttached(
            "Line", typeof(object), typeof(WordInlines),
            new PropertyMetadata(null, OnLineChanged));

    public static void SetLine(DependencyObject element, object? value) =>
        element.SetValue(LineProperty, value);

    public static object? GetLine(DependencyObject element) => element.GetValue(LineProperty);

    /// <summary>Word ranges for the currently rendered line, in RichTextBlock offset space.</summary>
    private static readonly DependencyProperty RangesProperty =
        DependencyProperty.RegisterAttached(
            "Ranges", typeof(object), typeof(WordInlines), new PropertyMetadata(null));

    private sealed record WordRange(int Start, int End, string Text, double Confidence);

    private static void OnLineChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is not RichTextBlock block) return;

        if (e.OldValue is CaptionLine previous)
        {
            previous.PropertyChanged -= OnLinePropertyChanged;
            _owners.Remove(previous);
        }

        if (e.NewValue is not CaptionLine line) return;

        block.SetValue(RangesProperty, null);
        line.PropertyChanged += OnLinePropertyChanged;
        _owners.Remove(line);
        _owners.Add(line, block);

        block.PointerMoved -= OnPointerMoved;
        block.PointerMoved += OnPointerMoved;
        block.PointerExited -= OnPointerExited;
        block.PointerExited += OnPointerExited;

        Render(block, line);
    }

    // Weak on the key, so a trimmed CaptionLine and its RichTextBlock become collectable
    // together. A plain dictionary here leaked one entry — and one visual subtree — per
    // finalised utterance, which matters in an app designed to run for hours.
    //
    // Deliberately no Unloaded hook to detach eagerly. Unloaded can fire for a block that is
    // still showing a current line (theme change, an ancestor collapsing), and detaching then
    // would unsubscribe PropertyChanged with nothing to re-attach it — the line would silently
    // stop updating, so a provisional caption would never upgrade to its final text. The weak
    // key already bounds growth without that risk, and the PropertyChanged handler is static,
    // so the subscription cannot keep the block alive either.
    private static readonly ConditionalWeakTable<CaptionLine, RichTextBlock> _owners = new();

    private static void OnLinePropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (sender is not CaptionLine line) return;
        if (e.PropertyName is not (nameof(CaptionLine.Text) or nameof(CaptionLine.Words))) return;
        if (_owners.TryGetValue(line, out var block)) Render(block, line);
    }

    private static void Render(RichTextBlock block, CaptionLine line)
    {
        try
        {
            var paragraph = new Paragraph();
            block.Blocks.Clear();
            block.Blocks.Add(paragraph);

            var words = line.Words;
            if (words is null || words.Count == 0)
            {
                // Provisional text, or a model that returned no word data.
                paragraph.Inlines.Add(new Run { Text = line.Text });
                block.SetValue(RangesProperty, null);
                return;
            }

            var ranges = new List<WordRange>(words.Count);
            foreach (var word in words)
            {
                // faster-whisper prefixes each word with the space that preceded it, so
                // styling the token would underline the gap before the word too.
                var text = word.Text;
                var start = 0;
                while (start < text.Length && char.IsWhiteSpace(text[start])) start++;
                var end = text.Length;
                while (end > start && char.IsWhiteSpace(text[end - 1])) end--;

                if (start > 0) paragraph.Inlines.Add(new Run { Text = text[..start] });

                var core = text[start..end];
                if (core.Length > 0)
                {
                    Inline inline;
                    var run = new Run { Text = core };
                    if (word.IsUncertain)
                    {
                        // Grey italic underline: three quiet signals rather than one loud one,
                        // noticeable when looked for and ignorable when reading. The underline
                        // takes the Foreground colour; there is no separate decoration brush.
                        var span = new Span
                        {
                            TextDecorations = Windows.UI.Text.TextDecorations.Underline,
                            FontStyle = Windows.UI.Text.FontStyle.Italic,
                            Foreground = UncertainBrush(),
                        };
                        span.Inlines.Add(run);
                        inline = span;
                    }
                    else
                    {
                        inline = run;
                    }
                    paragraph.Inlines.Add(inline);

                    // Read the offsets back from the element itself: derived character counts
                    // would drift from RichTextBlock's own offset space.
                    ranges.Add(new WordRange(
                        inline.ContentStart.Offset, inline.ContentEnd.Offset,
                        core, word.Confidence));
                }

                if (end < text.Length) paragraph.Inlines.Add(new Run { Text = text[end..] });
            }

            block.SetValue(RangesProperty, ranges);
        }
        catch
        {
            // A reading nicety must never take down an app someone relies on to follow a
            // conversation. Falling back to plain text loses only the marking.
            try
            {
                block.Blocks.Clear();
                var p = new Paragraph();
                p.Inlines.Add(new Run { Text = line.Text });
                block.Blocks.Add(p);
                block.SetValue(RangesProperty, null);
            }
            catch { /* leave whatever is on screen */ }
        }
    }

    private static void OnPointerMoved(object sender, PointerRoutedEventArgs e)
    {
        if (sender is not RichTextBlock block) return;
        if (block.GetValue(RangesProperty) is not List<WordRange> ranges || ranges.Count == 0) return;

        try
        {
            var point = e.GetCurrentPoint(block).Position;
            var pointer = block.GetPositionFromPoint(point);
            var offset = pointer.Offset;

            // ContentEnd is the position just past the last character, so the end is exclusive.
            // Inclusive on both sides would make adjacent words overlap at their shared
            // boundary and report whichever happened to be found first.
            var hit = ranges.FirstOrDefault(r => offset >= r.Start && offset < r.End);
            if (hit is null)
            {
                ToolTipService.SetToolTip(block, null);
                return;
            }

            // Rebuilding the tooltip on every move would flicker; only swap when the word does.
            if (ToolTipService.GetToolTip(block) is ToolTip existing &&
                existing.Content as string == Describe(hit)) return;

            ToolTipService.SetToolTip(block, new ToolTip { Content = Describe(hit) });
        }
        catch
        {
            // Hit testing is best-effort; a failed probe must not disturb the caption.
        }
    }

    private static string Describe(WordRange word) =>
        $"\u201c{word.Text}\u201d — {word.Confidence:0}% confident";

    private static void OnPointerExited(object sender, PointerRoutedEventArgs e)
    {
        if (sender is RichTextBlock block) ToolTipService.SetToolTip(block, null);
    }

    private static Brush? _uncertainBrush;

    /// <summary>
    /// Grey for uncertain words. Prefers the theme's secondary text brush so it tracks light
    /// and dark, but falls back to a fixed mid grey that reads on both — a missing resource
    /// key faults at render time as an opaque stowed exception, which is not worth risking
    /// for a styling detail.
    /// </summary>
    private static Brush UncertainBrush()
    {
        if (_uncertainBrush is not null) return _uncertainBrush;

        if (Application.Current?.Resources is { } resources &&
            resources.TryGetValue("TextFillColorSecondaryBrush", out var found) &&
            found is Brush themed)
        {
            return _uncertainBrush = themed;
        }

        return _uncertainBrush = new SolidColorBrush(
            new Windows.UI.Color { A = 255, R = 0x8A, G = 0x8A, B = 0x8A });
    }
}
