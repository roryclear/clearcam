/**
 * Device profile used to pick a starting model.
 *
 * Heavy runs at about half the speed of Full (BlazePose: Pixel 5 32/22 FPS lite/full,
 * MacBook Pro 2019 92/81/38 lite/full/heavy). Starting phones on Heavy meant several
 * slow seconds before the latency fallback stepped down, so 'auto' starts phones on Full.
 */

/**
 * Best-effort phone/tablet detection.
 * @param {Object} [env]
 * @param {Navigator} [env.navigator]
 * @param {(q: string) => {matches: boolean}} [env.matchMedia]
 */
export function isLikelyMobile({ navigator: nav = globalThis.navigator, matchMedia = globalThis.matchMedia } = {}) {
  if (!nav) return false;
  if (typeof nav.userAgentData?.mobile === 'boolean' && nav.userAgentData.mobile) return true;
  const ua = String(nav.userAgent || '');
  if (/Android|iPhone|iPad|iPod|Mobile/i.test(ua)) return true;
  // iPadOS reports a desktop Mac UA; it still has touch and a coarse pointer.
  const coarse = typeof matchMedia === 'function' && Boolean(matchMedia('(pointer: coarse)')?.matches);
  return coarse && (nav.maxTouchPoints ?? 0) > 1;
}

/**
 * Resolves the model setting to a concrete model.
 * @param {'auto'|'heavy'|'full'|'lite'} choice
 * @param {{ mobile: boolean }} profile
 */
export function resolveModelChoice(choice, { mobile }) {
  if (choice === 'heavy' || choice === 'full' || choice === 'lite') return choice;
  return mobile ? 'full' : 'heavy';
}
