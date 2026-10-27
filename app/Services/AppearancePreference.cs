namespace Sunno.Services;

/// <summary>Stable settings values, independent of XAML and future theme choices.</summary>
public static class AppearancePreference
{
    public const string System = "system";
    public const string Light = "light";
    public const string Dark = "dark";

    public static string Normalize(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        Light => Light,
        Dark => Dark,
        _ => System,
    };

    public static int IndexOf(string? value) => Normalize(value) switch
    {
        Light => 1,
        Dark => 2,
        _ => 0,
    };
}

public readonly record struct ClarityPalette(uint Good, uint Mid, uint Low);

/// <summary>Status text has its own contrast-safe palette in each appearance.</summary>
public static class AppearancePalette
{
    public static uint UncertainText(bool dark) => dark ? 0xFFD0D0D0 : 0xFF404040;

    public static ClarityPalette ForTheme(bool dark) => dark
        ? new(0xFF72CC98, 0xFFF0BE78, 0xFFF59E96)
        : new(0xFF216B42, 0xFF875100, 0xFFB42318);

    public static ClarityPalette ForHighContrast(uint windowText) =>
        new(windowText, windowText, windowText);
}
