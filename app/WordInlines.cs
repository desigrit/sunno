using System.Collections.Generic;
using System.Linq;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Media;
using Sunno.Models;

namespace Sunno;

/// <summary>
/// Renders a caption as individual word runs so uncertain words can be marked.
///
/// This has to be an attached property rather than a binding: styling individual words means
/// building <see cref="Inline"/>s, and XAML has no way to generate those from a collection.
///
/// The marking is deliberately quiet — a thin underline, no colour change. Someone reading
/// these to follow a conversation should be able to ignore the marks entirely and still read
/// normally; the signal is there for when a word looks wrong and they want to know whether
/// the model was unsure of it too.
/// </summary>
public static class WordInlines
{
    public static readonly DependencyProperty SourceProperty =
        DependencyProperty.RegisterAttached(
            "Source",
            typeof(object),
            typeof(WordInlines),
            new PropertyMetadata(null, OnSourceChanged));

    public static void SetSource(DependencyObject element, object? value) =>
        element.SetValue(SourceProperty, value);

    public static object? GetSource(DependencyObject element) =>
        element.GetValue(SourceProperty);

    private static void OnSourceChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is not TextBlock block) return;

        // No word data (a provisional line, or a model that returned none): leave the plain
        // Text binding alone rather than blanking the caption.
        if (e.NewValue is not IReadOnlyList<CaptionWord> words || words.Count == 0) return;

        try
        {
            block.Inlines.Clear();
            foreach (var word in words)
            {
                if (!word.IsUncertain)
                {
                    block.Inlines.Add(new Run { Text = word.Text });
                    continue;
                }

                // faster-whisper prefixes each word with the space that preceded it, so
                // styling word.Text wholesale underlines the gap before the word too. Split
                // the padding out and style only the word itself.
                var text = word.Text;
                var start = 0;
                while (start < text.Length && char.IsWhiteSpace(text[start])) start++;
                var end = text.Length;
                while (end > start && char.IsWhiteSpace(text[end - 1])) end--;

                if (start > 0)
                    block.Inlines.Add(new Run { Text = text[..start] });

                var core = text[start..end];
                if (core.Length > 0)
                {
                    // Italic and underlined in grey: three quiet signals rather than one loud
                    // one, so an uncertain word is noticeable when looked for and ignorable
                    // when reading. The underline takes the Foreground colour — there is no
                    // separate decoration brush — so the grey applies to both.
                    var span = new Span
                    {
                        TextDecorations = Windows.UI.Text.TextDecorations.Underline,
                        FontStyle = Windows.UI.Text.FontStyle.Italic,
                        Foreground = UncertainBrush(),
                    };
                    span.Inlines.Add(new Run { Text = core });
                    block.Inlines.Add(span);
                }

                if (end < text.Length)
                    block.Inlines.Add(new Run { Text = text[end..] });
            }

            var uncertain = words.Where(w => w.IsUncertain).ToList();
            ToolTipService.SetToolTip(block, uncertain.Count == 0
                ? null
                : "Less certain: " + string.Join(", ",
                    uncertain.Select(w => $"\u201c{w.Text.Trim()}\u201d {w.Confidence:0}%")));
        }
        catch
        {
            // A reading nicety must never take down an app someone is relying on to follow a
            // conversation. Falling back to plain text loses only the marking.
            try
            {
                block.Inlines.Clear();
                block.Text = string.Concat(words.Select(x => x.Text));
            }
            catch
            {
                // Nothing further to do; the caption stays as whatever it was.
            }
        }
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
