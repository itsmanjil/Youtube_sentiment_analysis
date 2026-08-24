// Shared sentiment palette used by every chart, badge, and legend so the
// same sentiment always wears the same color everywhere in the app.
//
// Hues are from the Wong (2011) colorblind-safe palette: the previous pure
// #FF0000/#0000FF/#008001 trio put red and green at similar lightness, which
// is indistinguishable for deuteranopic viewers; these three differ in both
// hue and lightness for the common color-vision deficiencies.
export const SENTIMENT_COLORS = {
  Positive: "#009E73",
  Neutral: "#0072B2",
  Negative: "#D55E00",
  // Withheld by the entropy gate — too uncertain to trust as a label.
  // Grey, not one of the three sentiment hues, so it reads as "no verdict"
  // rather than a fourth opinion.
  Uncertain: "#8A8A94",
};

// Light tint + dark ink pairs for badges, cards, and word-cloud panels.
// Dark-on-tint keeps text at readable contrast where white-on-saturated
// (small badge text) would fall below WCAG AA.
export const SENTIMENT_TINTS = {
  Positive: { bg: "#E6F5F0", border: "#B5E0D2", text: "#00543D" },
  Neutral: { bg: "#E8F1F8", border: "#B9D7EC", text: "#003D61" },
  Negative: { bg: "#FBEEE6", border: "#F2CDB3", text: "#7A3600" },
  Uncertain: { bg: "#EDEDF0", border: "#CFCFD6", text: "#45454D" },
};
