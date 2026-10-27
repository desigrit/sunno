using System;

namespace Sunno.Services;

/// <summary>Size the list without letting an expanding model section consume the speakers.</summary>
public static class ModelPickerLayout
{
    public static double MaximumListHeight(double sidebarHeight, double sidebarPadding,
                                          double headerHeight, double sectionMargin)
    {
        return Math.Max(0, Math.Max(0, sidebarHeight - sidebarPadding) * 0.75
                           - headerHeight - sectionMargin);
    }
}
