---
version: alpha
name: RetineX
description: Design system for RetineX, a visual accessibility tool that turns an image into a written description. Calm editorial product design, restrained sage and green palette, hyperlegible type.
colors:
  canvas: "#F3F2EC"
  surface: "#FFFFFF"
  neutral: "#DAD7CD"
  secondary: "#A3B18A"
  accent: "#588157"
  primary: "#3A5A40"
  on-surface: "#344E41"
  on-surface-muted: "#5A6B55"
  error: "#8C3A2E"
  warning: "#7A5A14"
typography:
  display-xl:
    fontFamily: Avegas Royale
    fontSize: 72px
    fontWeight: 400
    lineHeight: 1.04
    letterSpacing: -0.02em
  display-lg:
    fontFamily: Avegas Royale
    fontSize: 48px
    fontWeight: 400
    lineHeight: 1.1
    letterSpacing: -0.015em
  h2:
    fontFamily: Avegas Royale
    fontSize: 32px
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: -0.01em
  h3:
    fontFamily: Atkinson Hyperlegible Next
    fontSize: 21px
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: 0em
  body-lg:
    fontFamily: Atkinson Hyperlegible Next
    fontSize: 19px
    fontWeight: 400
    lineHeight: 1.65
    letterSpacing: 0em
  body-md:
    fontFamily: Atkinson Hyperlegible Next
    fontSize: 17px
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: 0em
  label:
    fontFamily: Atkinson Hyperlegible Next
    fontSize: 13px
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: 0.08em
  button:
    fontFamily: Atkinson Hyperlegible Next
    fontSize: 16px
    fontWeight: 700
    lineHeight: 1
    letterSpacing: 0.01em
  caption:
    fontFamily: Atkinson Hyperlegible Next
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0.01em
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 40px
  2xl: 64px
  3xl: 96px
  4xl: 160px
rounded:
  none: 0px
  sm: 2px
  md: 4px
  lg: 8px
  full: 9999px
components:
  page:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.on-surface}"
    typography: "{typography.body-md}"
  panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.md}"
    padding: "{spacing.lg}"
  meta-text:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.on-surface-muted}"
    typography: "{typography.caption}"
  section-inverse:
    backgroundColor: "{colors.on-surface}"
    textColor: "{colors.neutral}"
    padding: "{spacing.3xl}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    padding: "{spacing.md}"
  button-primary-hover:
    backgroundColor: "{colors.on-surface}"
    textColor: "{colors.surface}"
  button-secondary:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.primary}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    padding: "{spacing.md}"
  button-subtle:
    backgroundColor: "{colors.neutral}"
    textColor: "{colors.on-surface}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
  button-disabled:
    backgroundColor: "{colors.neutral}"
    rounded: "{rounded.md}"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    typography: "{typography.body-md}"
    rounded: "{rounded.md}"
    padding: "{spacing.sm}"
  input-error:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.error}"
    typography: "{typography.caption}"
  alert-warning:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.warning}"
    typography: "{typography.caption}"
  divider:
    backgroundColor: "{colors.neutral}"
    height: 1px
  rule-strong:
    backgroundColor: "{colors.secondary}"
    height: 2px
  focus-ring:
    backgroundColor: "{colors.accent}"
    height: 2px
---

# RetineX Design System

Canonical visual source of truth for all RetineX frontend work. If an implementation choice is not covered here, follow the nearest rule by analogy rather than inventing a new pattern, then add the rule back to this file.

## Overview

**Product.** RetineX turns an image into a written description and, in a later stage, spoken narration, so that someone who cannot see an image can still know what is in it. The interface is written for that person. Where a capability is still being built, the interface says so plainly in product language and never implies that it works.

**Direction.** Calm editorial product design with premium SaaS level polish: generous whitespace, strong typography, restrained green and sage color, subtle motion, and accessibility first interaction. Text is the primary material, hairline rules do the work that shadows would do elsewhere, and the layout reads as edited rather than assembled.

**The tradeoff this makes.** This direction gives up visual novelty and high energy. It buys clarity, durability, and credibility, which is the right trade for a product whose whole job is to be read. If a future page needs to feel thrilling, this system is the wrong tool and the decision should be revisited in writing, not worked around in a stylesheet.

**Personality.** Calm, intelligent, dependable, sophisticated, accessible, human, research driven.

**Visual quality reference.** Passionfroot (https://www.passionfroot.me/) is the bar for hierarchy, whitespace, typographic scale, and interaction polish only. Do not copy its branding, logo, copy, layouts, illustrations, or component designs. RetineX looks nothing like it.

**Non negotiable.** Accessibility is the product, so it is also the design constraint. A choice that looks better but reads worse loses.

## Colors

Every color is named after something real rather than picked from a palette generator. The five brand colors are fixed and supplied; they are not up for reinterpretation.

| Token | Hex | Name and referent | Job |
| --- | --- | --- | --- |
| `neutral` | `#DAD7CD` | Dust Grey, lichen on limestone | Dividers, hairlines, sunken fills, disabled surfaces, text on dark sections |
| `secondary` | `#A3B18A` | Dry Sage, the silver underside of a sage leaf | Emphasis rules, decorative marks, never body text on light |
| `accent` | `#588157` | Fern, a new bracken frond | Focus rings, active state, non text marks only |
| `primary` | `#3A5A40` | Hunter Green, waxed cotton canvas | Primary action, links, the one loud color |
| `on-surface` | `#344E41` | Pine Teal, conifer needle in shade | Primary text, and the background of inverted sections |

Two further values are **approved supporting UI colors**, derived from the same family because the five brand colors do not cover every role. They exist for readability and accessibility, and they are as binding as the five above.

| Token | Hex | Why it exists |
| --- | --- | --- |
| `canvas` | `#F3F2EC` | Dust Grey lightened to a paper tint. Dust Grey at full strength is too heavy for a full page background and leaves no room for a lighter surface above it. |
| `on-surface-muted` | `#5A6B55` | A desaturated mid tone between Fern and Hunter Green. The supplied palette has no value that is both lighter than Pine Teal and still passes AA for secondary text. |

`surface` is `#FFFFFF`, used for panels that sit above the canvas.

**Measured contrast.** Computed, not estimated. Ratios against `canvas` `#F3F2EC`: `on-surface` 8.10, `primary` 6.89, `on-surface-muted` 5.10, `accent` 3.99, `secondary` 2.03. Against `#FFFFFF`: `on-surface` 9.08, `primary` 7.73, `accent` 4.48.

Three consequences are binding:

- **Fern is never text.** At 4.48 on white it misses AA by a hair. It is a focus ring, an active underline, a marker, and nothing else.
- **Dry Sage is never text on a light background.** At 2.03 it is decorative only. On a Pine Teal section it reaches 3.98, which is large text and UI only.
- **Primary buttons darken on hover, they do not brighten.** Hunter Green to Pine Teal keeps white text at 9.08. Hunter Green to Fern would drop it to 4.48 and fail.

**Inverted sections.** Pine Teal `#344E41` as background with Dust Grey `#DAD7CD` text reaches 6.31 and is the only approved dark treatment. Use it for at most one band per page.

**There is no dark mode.** The primary experience is light, and that is a decision rather than an omission. Do not add a dark theme, a theme toggle, or `prefers-color-scheme` overrides at this stage. Inverted sections above are the full extent of dark surfaces.

**Semantic colors.** Derived from the same earth family so they belong to the brand: `error` `#8C3A2E` (iron oxide, 6.78 on canvas), `warning` `#7A5A14` (raw umber, 5.67 on canvas). Success reuses `primary`; a separate success green next to a green brand would be noise. Do not add further semantic colors without adding them here first.

**Accent scarcity.** `primary` carries one meaning: this is the action. Keep it under roughly 5 percent of the pixels on any screen. Everything else does its work with weight, size, spacing, and value.

## Typography

Two families. Not three. The pairing is deliberate: Avegas Royale carries the RetineX personality, Atkinson Hyperlegible Next carries readability. Neither is a placeholder. Do not add or download font files beyond what is described here.

**Avegas Royale** is the display and brand face, used for the wordmark lockup, page titles, and section titles. It is a licensed font and is **not currently in the repository**. Expected setup, to be completed when the files are supplied:

- Place licensed `.woff2` files in `app/static/fonts/avegas-royale/`.
- Declare with `@font-face`, `font-display: swap`, and the fallback stack `"Avegas Royale", "Iowan Old Style", Georgia, serif`.
- Never substitute a lookalike from a font CDN, and never ship the files to a public repository unless the license allows redistribution. License terms are NOT VERIFIED; confirm before any deployment.
- Until the files arrive, display type renders in the fallback. That is expected and is not a bug to patch with a different font.

**Atkinson Hyperlegible Next** is the body, UI, and navigation face. It was drawn by the Braille Institute specifically so that similar letterforms stay distinguishable at low vision, which is the argument for using it on a product built for visual accessibility. Available from Google Fonts; self host rather than hotlink. Its exact license is NOT VERIFIED, confirm before deployment. Fallback stack: `"Atkinson Hyperlegible Next", "Atkinson Hyperlegible", system-ui, sans-serif`.

**Roles.** `display-xl` for the one statement at the top of a page. `display-lg` for page titles. `h2` for section titles. `h3` for subsections, and note that `h3` switches to Atkinson, because below roughly 24px the display face costs more in legibility than it returns in character. `body-lg` for lead paragraphs and `body-md` for everything else, both at a line length of 62 to 74 characters. `label` is uppercase with wide tracking, used for eyebrows and form labels. `caption` is for metadata, figure captions, and research footnotes. `button` is its own token so buttons never inherit a paragraph size.

**Rules.** One display face per screen in at most two sizes. No italic in UI chrome. No letter spacing on body text. Numerals in research tables are tabular. A mono family is the single permitted future addition, and only when the Workspace needs to show feature dimensions or BLEU tables; add it here before using it.

## Layout

**Spacing.** A 4px base with a deliberately gapped scale: 4, 8, 16, 24, 40, 64, 96, 160. Any value not on this scale is a mistake. Vertical rhythm between major sections uses `3xl` or `4xl`; inside a component use `sm` through `lg`.

**Whitespace is the primary design tool here.** When a layout feels wrong, the fix is almost always more space and less ornament. Section padding should feel slightly too generous.

**Content widths.** Prose columns cap at 68ch. The main content column caps at 1120px. Full bleed is reserved for inverted bands and the image upload surface. Never let a text column run the full width of a desktop display.

**Grid.** 12 columns on desktop, 8 on tablet, 4 on mobile, with a 24px gutter. Align to the grid and keep a single strong left edge down the page. Centered layouts are for the hero statement only.

**Responsive.** Mobile first, with breakpoints at 640px, 1024px, and 1440px. Type scales down roughly 20 percent below 640px, with `display-xl` dropping to 40px. Spacing drops one step on the scale, so `3xl` becomes `2xl`. Touch targets are at least 44 by 44px. Navigation collapses to a disclosure button below 1024px, and never to a bottom tab bar on the public site. Tables in the Workspace scroll horizontally inside their own container rather than reflowing into cards.

## Elevation & Depth

**There are effectively no shadows.** Depth comes from value, hairlines, and space, in that order. A panel separates from the canvas because it is `#FFFFFF` against `#F3F2EC`, not because it floats.

Exactly one shadow is permitted, and only for elements that genuinely sit above the page in a temporary way, meaning dropdowns, popovers, and modals: `0 2px 8px rgba(52, 78, 65, 0.10)`. It is tinted with Pine Teal rather than black, because a neutral black shadow over a warm canvas reads as grey dirt.

Forbidden: glassmorphism, backdrop blur as decoration, glow, neon borders, stacked shadow layers, and shadows on cards, buttons, or inputs in their resting state.

## Shapes

Radius is small and hierarchical, and it signals permanence: the more fixed a thing is, the squarer it is.

- `none` `0px` for page sections, inverted bands, tables, and image frames.
- `sm` `2px` for hairline chips and tags.
- `md` `4px` for inputs and panels. This is the default.
- `lg` `8px` for buttons, modals, and the image upload drop zone. Buttons were raised from `md` to `lg` on 2026-10-13: at a 44px target height a 4px corner reads as unfinished, and 8px is the smallest radius that reads as deliberate. It is two steps below `full`, so a button is still visibly not a pill.
- `full` is reserved for a status dot and a user avatar. Nothing else is ever a pill.

Borders are 1px of `neutral`, or 2px of `secondary` when a rule needs real emphasis. A border is the default way to bound a thing; a fill is the exception.

## Components

**Buttons.** Four kinds and no more. `button-primary` is Hunter Green with white text and is used once per view for the single most important action. `button-secondary` is canvas with a 1px `neutral` border and Hunter Green text. `button-subtle` is a Dust Grey fill with Pine Teal text, for low stakes repeated actions. A text only button is permitted for destructive confirmation, in `error`. All use `lg` radius, 12px by 20px padding, and the `button` type token. Hover darkens the fill one step and never moves the button. Active darkens again and may shift 1px down. Focus shows a 2px Fern ring at 2px offset, always visible, never removed. Disabled uses the Dust Grey fill at full opacity with `on-surface-muted` text and `cursor: not-allowed`; do not communicate disabled state with opacity alone. No gradients on any button.

**Inputs and forms.** White fill, 1px `neutral` border, `md` radius, 12px padding, `body-md` text. The border darkens to `primary` on focus and the Fern focus ring appears alongside it. Labels sit above the field in the `label` token and are always present; placeholder text is never a substitute for a label. Errors show `error` text below the field plus an `error` border, and state the fix rather than the failure. Required fields are marked in the label text, not with a bare asterisk. Every input has a programmatically associated `<label>`.

**Cards and surfaces.** A card has to earn itself. Use one only when several items are genuinely parallel and need visual separation, such as the three team members. Prose, a hero, a single statistic, or a form is not a card. When a card exists it is white on canvas, `md` radius, `lg` padding, 1px `neutral` border, no shadow. Never nest a card inside a card. The default answer to any layout question is open space with a hairline rule, not another container.

**Navigation.** A single horizontal bar, canvas background, one hairline `neutral` rule beneath it, no shadow, no blur, no shrinking on scroll. The logo lockup sits left, links sit right in `body-md`. The current page is marked with a 2px Fern underline, not a filled pill. There is exactly one underline in the bar and it is shared: it glides horizontally to whichever link is hovered or focused, and returns to the current page when the pointer and focus leave. It is a progressive enhancement, so without JavaScript the current page keeps a plain static underline and no second marker is ever drawn. One `button-primary` is permitted in the bar for the main entry action. Below 1024px the links collapse into a disclosure button that opens a full height panel; the trigger carries `aria-expanded` and focus is trapped while it is open. The footer repeats the navigation plus Privacy Policy and Terms, in `caption`.

## Do's and Don'ts

These are hard rules. A pull request that breaks one is wrong even if it looks good.

**Never use:** purple or blue to violet gradients; the generic AI blue and violet aesthetic; pill shapes outside the two permitted uses; fake testimonials, reviews, metrics, counters, or user numbers; emoji as icons; em dashes in any interface copy; AI generated imagery; vague hero copy; glassmorphism; heavy or stacked shadows; large radius cards everywhere; scroll triggered animation on most elements; decoration with no purpose; random gradients; floating blobs; unnecessary floating elements.

**Do:** delete the element you are unsure about; prefer a hairline to a container and a container to a shadow; let a section breathe before adding anything to it; keep `primary` scarce; write the contrast ratio into the review when you introduce any new color pair.

**Specific to this project:** never present an unfinished research stage as working. The pipeline, which runs image to VGG16 to a 4096 dimensional representation to TF-IDF to K-Means pseudo classes to Fisher Score to 512 selected features to LSTM to BLEU evaluation to narration, is partly unimplemented. No BLEU score, accuracy figure, latency number, or capability claim appears in the interface until it is measured in this repository. Placeholders read as placeholders, for example "Not yet evaluated", never as a plausible number.

## Motion

Motion is quiet and explains something. If it does not aid comprehension, remove it.

**Permitted:** opacity fades; fade up with a maximum 12px translate on first reveal of a section, once, not on every scroll; color and border transitions on hover and focus; a height transition on a disclosure; a restrained page transition; a transform transition on the single shared navigation underline as it glides between links.

**The hero entrance** is the one place a stagger is used: the eyebrow, the headline, and the supporting copy with its action row fade up in that order, three items at 70ms apart, each a 320ms fade up of 10px. It reuses the page transition's own keyframe rather than defining a second one, so the hero and the page rise by the same amount. It is a plain CSS animation on load, so it needs no script and can never leave content hidden; it stands down while a page transition is running, because the transition is already that page's entrance. A section reveal below the fold is the same 320ms fade up of 12px, triggered once by an IntersectionObserver, and the content is fully visible if that observer never runs.

**Timing.** 120ms for hover and focus, 200ms for a disclosure or a panel, 320ms for a section reveal. Easing is `cubic-bezier(0.22, 0.61, 0.36, 1)`, which is a plain ease out. Nothing exceeds 400ms.

**Page transitions.** Moving between two pages is one crossfade: the outgoing page fades out over 120ms while the incoming page fades in over 260ms and rises 10px, so the whole exchange is 260ms. The earlier cap of 200ms was raised to 260ms on 2026-10-13, because a crossfade carrying a translate reads as a snap below roughly 240ms and the extra 60ms is what makes it read as deliberate instead. It is still well inside the 400ms ceiling. The header and the navigation underline are excluded from that movement and carry their own names, so the bar stays put and the underline glides from the old page's link to the new one instead of the whole chrome lifting. This is implemented with the browser's cross-document View Transitions API through a single `@view-transition` rule, so no script touches a link and a browser without the API simply navigates.

**Forbidden:** bounce, spring physics, scale above 1.02, parallax, staggered cascades of more than three items, animated gradients, flashy loaders, and anything that animates on every element in a list. Never animate the logo.

**Loading.** A static label such as "Generating description" plus a thin indeterminate bar in `accent`. No spinners that pulse, no skeleton shimmer.

**Reduced motion is mandatory.** Under `prefers-reduced-motion: reduce`, transforms and reveals are disabled, elements appear in their final state, and transitions drop to at most an 80ms opacity change. Content must never require motion to be understood.

## Accessibility

This is an accessibility product, so these are requirements and not goals.

- WCAG 2.2 AA minimum: 4.5:1 for body text, 3:1 for large text and UI boundaries. Every pair in this file is measured above.
- Semantic HTML first. One `h1` per page, headings in order, real `<button>` and `<a>` elements, landmarks for header, nav, main, and footer.
- Every interactive element is reachable and operable by keyboard in a logical order. The focus ring is a 2px Fern outline at 2px offset and is never removed. A skip to content link is the first focusable element.
- Form controls have associated labels, errors are announced, and the field with the error receives focus.
- Images carry meaningful alt text. Decorative images use `alt=""`. A generated description shown in the Workspace is page content, not alt text, and lives in a live region so a screen reader announces it when it arrives.
- Audio narration never autoplays. It has visible play, pause, and stop controls and a text transcript that is always present, because the text is the primary output and the audio is a second channel.
- Target size at least 44 by 44px. Text resizes to 200 percent without loss of content. Respect `prefers-reduced-motion` and `prefers-contrast`.
- Color never carries meaning alone; pair it with text, an icon, or a border.

## Iconography

One icon set, stroke based, 1.5px stroke at 20 or 24px, square cap, matching the restrained geometry. Lucide is the working default; pick one set and do not mix. Icons are secondary to typography and are never decorative. Each icon either has a visible text label or an `aria-label`, and a standalone icon button always has an accessible name. Emoji are never icons.

## Imagery

The only images in the product are real: the supplied brand assets, real photographs of the team when they are provided, real screenshots of the running application, and user uploaded images inside the Workspace. No AI generated imagery, no stock photography standing in for a product, no fabricated screenshots, no illustrated mascots. A user uploaded image is displayed in a square cornered frame with a 1px `neutral` border at its natural aspect ratio, never cropped to a decorative shape.

## Brand Assets

The supplied files are the source of truth and are never recreated, redrawn, recolored, or regenerated.

- **Logo:** `app/static/brand/retinex-logo.png`, a transparent PNG of 813 by 307, an open eye mark plus the wordmark, with the tagline "Turning Vision Into Voice". The wordmark is black, the eye mark and the accent letterforms are `#365043`. As exported it carries 34px of fully transparent padding on its left edge, which at render size pushed the visible wordmark about 7px to the right of the content column and left the navigation looking misaligned against the page below it. `scripts/build_favicon.py` therefore also crops it to its own alpha bounds and writes `app/static/brand/retinex-logo-flush.png`, 732 by 159, which is what the navigation renders at 170px wide. Only fully transparent pixels are removed; no visible pixel, proportion or colour changes, and the supplied file is never written to. An SVG version is still worth requesting from whoever produced it, since a PNG wordmark will not stay crisp at larger sizes.
- **Favicon:** supplied as `app/static/brand/favicon-source.png`, a transparent PNG of 1414 by 706, the eye mark alone in cream `#E9E5D9`. It is not square, so it is not usable as a favicon directly. `scripts/build_favicon.py` crops it to its own alpha bounds and centres it on a 512 by 512 transparent canvas, producing two served variants:
  - `favicon.png`, refilled with Pine Green `#324A3D`, for light browser chrome. It is also the fallback for browsers that ignore `media` on a link element, and the Apple touch icon, because touch icons are composited on light backgrounds.
  - `favicon-dark.png`, the cream master's own colour, for dark browser chrome.

  Two variants exist because a favicon sits in browser chrome rather than on the page, and no single value is visible on both. Measured: cream on Chrome's light tab bar `#DEE1E6` is 1.04:1, which is invisible; Pine Green on the same surface is 7.34:1. On a dark tab bar `#35363A` cream reaches 9.58:1 and Pine Green drops to 1.25:1, which is why the pair is needed rather than one value. This is browser chrome only and does not give the site a dark theme. The green variant changes the fill colour and nothing else: every alpha value, and therefore the shape and its antialiasing, comes unchanged from the master, which is never written to.
- **Clear space** around the logo is at least the height of the eye mark on all sides. Minimum width is 160px. Do not stretch, rotate, add effects to, or place the logo on a busy background.

**The brand assets were re-supplied in the project's own greens on 2026-10-12, replacing the earlier electric blue `#0039FD` artwork.** The blue is no longer present in either asset and is no longer part of the brand. The replacements are transparent, so both sit flush on any surface.

The asset colours are close to the palette but are not the same tokens, and they are not interchangeable with it: the logo's mark is `#365043`, the favicon's green variant is `#324A3D`, while `on-surface` is `#344E41`. Do not substitute a token for an asset colour or an asset colour for a token. Supplied artwork is still never redrawn or regenerated; the only permitted derivations are the crop, pad, resize and single-colour refill performed by `scripts/build_favicon.py`, which writes its own output files and never overwrites a supplied master.

## Content & Copy

Concrete, concise, human, factual. Say what the thing does and who it is for.

Never write "The future of AI", "Revolutionizing visual intelligence", "Empowering the future", "Next generation AI", or any sentence that would survive unchanged on a different product's website. No em dashes. Sentence case for headings and buttons. Buttons name their action, so "Generate description" and not "Submit" or "Get started". Error messages say what to do next.

Never fabricate users, customers, reviews, testimonials, statistics, metrics, accuracy figures, BLEU scores, performance improvements, or research findings. If a number is not measured in this repository, it does not appear. Use a visible placeholder and ask for the real value.

The team is Bishr Muhammad Iqbal, Yaseen Khan, and Fateh Mohammed Khan. Do not invent titles, biographies, credentials, achievements, or responsibilities for them. Those will be supplied.

## Implementation Rules

For whoever, human or agent, builds the frontend next.

1. Read this file before writing any UI. It overrides personal preference and framework defaults.
2. Emit these tokens once as CSS custom properties or a Tailwind theme, generated from the frontmatter above. Never hardcode a hex value, a font size, or a spacing number in a component.
3. If a needed value is missing, add it to this file first, with its reason and its measured contrast, then use it. Do not add it inline and plan to tidy later.
4. Reuse before you add. A new component needs a reason that an existing one cannot cover.
5. Run `npx @google/design.md lint design.md` after editing this file. Zero errors, and no unexplained warnings.
6. Check every new color pair for contrast before committing it.
7. Keep the research integrity rule above in mind on every page that touches results.
8. The planned surfaces are Home, How RetineX Works, The Minds Behind RetineX, Privacy Policy, Terms and Conditions, Sign In and Sign Up, and the RetineX Workspace. They share this system. None of them exist yet.

## Settled Decisions

Approved by the project owner. Do not reopen these without an explicit instruction.

- **Direction** is calm editorial product design with premium SaaS level polish. Passionfroot is the quality reference only.
- **Logo blue** stays inside the supplied assets and never enters the UI palette.
- **Derived colors** `canvas` `#F3F2EC` and `on-surface-muted` `#5A6B55` are approved supporting UI colors.
- **Typography** is Avegas Royale for display and Atkinson Hyperlegible Next for body and UI.
- **No dark mode** at this stage. The primary experience is light. The favicon's `prefers-color-scheme` variants are browser chrome, not a site theme, and do not reopen this.

## Outstanding Assets

Not decisions, just files that are still needed. Neither blocks wireframing.

1. **Avegas Royale font files and license.** Not in the repository. License terms are NOT VERIFIED. Both are needed before any public deployment. Until then, display type falls back as described in Typography.
2. **An SVG version of the logo.** A PNG wordmark will not stay crisp above its natural 813px width. The square favicon export that used to be listed here is done: `scripts/build_favicon.py` generates both variants from the supplied master.
