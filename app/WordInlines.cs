using System.Collections.Generic;
using System.Linq;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
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

                // Underline via a Span wrapping the run. Deliberately no Foreground override
                // and no per-inline tooltip: a Span is not a FrameworkElement, so
                // ToolTipService cannot attach to it, and theme brushes are not reliably
                // resolvable from Application.Current.Resources. Both of those crashed the
                // app with an opaque stowed exception. The scores go on the TextBlock.
                var span = new Span { TextDecorations = Windows.UI.Text.TextDecorations.Underline };
                span.Inlines.Add(new Run { Text = word.Text });
                block.Inlines.Add(span);
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
}
