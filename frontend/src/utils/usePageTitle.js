import { useEffect } from "react";

const BASE_TITLE = "YouTube Sentiment Analysis";

// Sets the document title for a route, restoring the plain app name for
// routes that don't pass one. Keeps browser tabs/history distinguishable.
export default function usePageTitle(title) {
  useEffect(() => {
    document.title = title ? `${title} · ${BASE_TITLE}` : BASE_TITLE;
  }, [title]);
}
