import { Banner } from "../components/Layout";
import { Card, ErrorNotice, Skeleton } from "../components/ui";
import { formatTtl } from "../lib/format";
import { useDiscern } from "../state/DiscernContext";

const REPO_URL = "https://github.com/VeerbhadraMahant/Discern#readme";

export function AboutView() {
  const { info, connection, retryConnect } = useDiscern();
  return (
    <>
      <Banner title="ABOUT" kicker="What Discern stores, which models it uses, and what is measured." />
      <div className="mx-auto grid max-w-[1440px] grid-cols-1 gap-10 px-4 py-10 md:px-8 lg:grid-cols-2">
        {connection.status === "unreachable" && (
          <div className="lg:col-span-2">
            <ErrorNotice error={connection.error} onRetry={retryConnect} />
          </div>
        )}
        <Card as="section" aria-labelledby="stored-h" className="flex flex-col gap-3">
          <h2 id="stored-h" className="font-display text-heading-sm leading-[0.95] tracking-[-0.04em]">
            What is stored
          </h2>
          {info ? (
            <>
              <p>
                Your upload and its results are kept for {formatTtl(info.limits.ttl_seconds)}, then deleted. Press Start over to delete them
                sooner.
              </p>
              <p>
                Your media is kept longer only if you tick the box on the Feedback tab. It is unticked by default.
              </p>
              <p className="num">
                Limits: {info.limits.max_upload_mb} MB per file, videos up to {info.limits.max_video_seconds} seconds, {info.limits.max_queries} questions per
                session, {info.limits.max_pixels.toLocaleString("en-US")} pixels per image.
              </p>
            </>
          ) : (
            <Skeleton className="h-24 w-full" />
          )}
        </Card>

        <Card as="section" aria-labelledby="meas-h" className="flex flex-col gap-3">
          <h2 id="meas-h" className="font-display text-heading-sm leading-[0.95] tracking-[-0.04em]">
            Measured so far
          </h2>
          {info ? (
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2">
              <dt className="font-semibold">Profile</dt>
              <dd className="font-normal">{info.profile}</dd>
              <dt className="font-semibold">GPU seconds</dt>
              <dd className="num font-normal">{info.measured_gpu_seconds === null ? "not measured yet" : info.measured_gpu_seconds}</dd>
              <dt className="font-semibold">Memory version</dt>
              <dd className="font-normal">{info.memory_version ?? "none"}</dd>
            </dl>
          ) : (
            <Skeleton className="h-24 w-full" />
          )}
        </Card>

        <Card as="section" aria-labelledby="models-h" className="flex flex-col gap-3 lg:col-span-2">
          <h2 id="models-h" className="font-display text-heading-sm leading-[0.95] tracking-[-0.04em]">
            Models
          </h2>
          {info ? (
            <>
              <p>Licenses are shown exactly as the app reports them. Some models are licensed for non-commercial use only, so check the license text before reusing a model elsewhere.</p>
              <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Scrollable table">
                <table className="w-full min-w-[480px] border-collapse text-left">
                  <caption className="sr-only">Models and licenses</caption>
                  <thead>
                    <tr className="border-b border-ink">
                      <th scope="col" className="py-2 pr-4 font-semibold">Role</th>
                      <th scope="col" className="py-2 pr-4 font-semibold">Model</th>
                      <th scope="col" className="py-2 font-semibold">License</th>
                    </tr>
                  </thead>
                  <tbody>
                    {info.models.map((m, i) => (
                      <tr key={i} className="border-b border-ink/40">
                        <td className="py-2 pr-4 font-normal">{m.role}</td>
                        <td className="py-2 pr-4 font-normal">{m.name}</td>
                        <td className="py-2 font-normal">{m.license}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <Skeleton className="h-24 w-full" />
          )}
        </Card>

        <p className="lg:col-span-2">
          <a className="link" href={REPO_URL} target="_blank" rel="noreferrer">
            Read the repository README
          </a>
        </p>
      </div>
    </>
  );
}
