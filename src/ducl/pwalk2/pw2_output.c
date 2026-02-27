/*
 * pw2_output.c — CSV formatting, output buffering, exclude list for pwalk2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <linux/stat.h>
#include "pwalk2.h"

/* --- CSV escape: double quotes are doubled per RFC 4180 --- */
void
pw2_csv_escape(const char *in, char *out, size_t outsize)
{
    char *end = out + outsize - 1;

    while (*in && out < end) {
        if (*in == '"') {
            if (out + 1 < end) {
                *out++ = '"';
                *out++ = '"';
            }
            in++;
        } else {
            *out++ = *in++;
        }
    }
    *out = '\0';
}

/* --- Flush thread-local output buffer under mutex --- */
void
pw2_flush_output(struct pw2_worker *w)
{
    if (w->outbuf_used > 0) {
        pthread_mutex_lock(&g_state.output_mutex);
        fwrite(w->outbuf, 1, w->outbuf_used, stdout);
        pthread_mutex_unlock(&g_state.output_mutex);
        w->outbuf_used = 0;
    }
}

static void
buffer_output(struct pw2_worker *w, const char *data, int len)
{
    if (w->outbuf_used + len > PW2_OUTBUF_SIZE) {
        pw2_flush_output(w);
        if (len > PW2_OUTBUF_SIZE) {
            pthread_mutex_lock(&g_state.output_mutex);
            fwrite(data, 1, len, stdout);
            pthread_mutex_unlock(&g_state.output_mutex);
            return;
        }
    }
    memcpy(w->outbuf + w->outbuf_used, data, len);
    w->outbuf_used += len;
}

/* Format a non-directory file as CSV */
void
pw2_format_file(struct pw2_worker *w, const char *path,
                const char *exten, struct statx *stx,
                ino_t parent_inode, int depth)
{
    char fname[FILENAME_MAX * 2];
    char exten_csv[FILENAME_MAX];
    char line[FILENAME_MAX * 4];
    int len;

    pw2_csv_escape(path, fname, sizeof(fname));
    if (exten)
        pw2_csv_escape(exten, exten_csv, sizeof(exten_csv));
    else
        exten_csv[0] = '\0';

    len = snprintf(line, sizeof(line),
        "%llu,%llu,%d,\"%s\",\"%s\",%u,%u,%lld,%lu,%llu,%u,\"%07o\",%lld,%lld,%lld,%ld,%ld\n",
        (unsigned long long)stx->stx_ino,
        (unsigned long long)parent_inode,
        depth,
        fname, exten_csv,
        (unsigned)stx->stx_uid,
        (unsigned)stx->stx_gid,
        (long long)stx->stx_size,
        (unsigned long)makedev(stx->stx_dev_major, stx->stx_dev_minor),
        (unsigned long long)stx->stx_blocks,
        (unsigned)stx->stx_nlink,
        (unsigned)stx->stx_mode,
        (long long)stx->stx_atime.tv_sec,
        (long long)stx->stx_mtime.tv_sec,
        (long long)stx->stx_ctime.tv_sec,
        (long)-1, (long)0);

    if (len > 0 && len < (int)sizeof(line))
        buffer_output(w, line, len);
}

/* Format a directory entry as CSV (with file count and size sum) */
void
pw2_format_dir(struct pw2_worker *w, const char *path,
               const char *exten, struct statx *stx,
               ino_t parent_inode, int depth,
               long file_count, long dir_size)
{
    char fname[FILENAME_MAX * 2];
    char exten_csv[FILENAME_MAX];
    char line[FILENAME_MAX * 4];
    int len;

    pw2_csv_escape(path, fname, sizeof(fname));
    if (exten)
        pw2_csv_escape(exten, exten_csv, sizeof(exten_csv));
    else
        exten_csv[0] = '\0';

    /* Directory depth is depth - 1 to match pwalk convention */
    len = snprintf(line, sizeof(line),
        "%llu,%llu,%d,\"%s\",\"%s\",%u,%u,%lld,%lu,%llu,%u,\"%07o\",%lld,%lld,%lld,%ld,%ld\n",
        (unsigned long long)stx->stx_ino,
        (unsigned long long)parent_inode,
        depth - 1,
        fname, exten_csv,
        (unsigned)stx->stx_uid,
        (unsigned)stx->stx_gid,
        (long long)stx->stx_size,
        (unsigned long)makedev(stx->stx_dev_major, stx->stx_dev_minor),
        (unsigned long long)stx->stx_blocks,
        (unsigned)stx->stx_nlink,
        (unsigned)stx->stx_mode,
        (long long)stx->stx_atime.tv_sec,
        (long long)stx->stx_mtime.tv_sec,
        (long long)stx->stx_ctime.tv_sec,
        file_count, dir_size);

    if (len > 0 && len < (int)sizeof(line))
        buffer_output(w, line, len);
}

/* --- Exclude list --- */

void
pw2_load_exclude_list(const char *fname)
{
    FILE *fp;
    char buf[PATH_MAX];

    fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "pwalk2: could not open exclude file: %s\n", fname);
        exit(1);
    }

    while (fgets(buf, sizeof(buf), fp) && g_state.exclude_count < PW2_MAXEXFILES) {
        size_t len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n')
            buf[--len] = '\0';
        if (len == 0)
            continue;
        g_state.exclude_paths[g_state.exclude_count] = strdup(buf);
        g_state.exclude_count++;
    }
    fclose(fp);
}

int
pw2_check_exclude(const char *path)
{
    for (int i = 0; i < g_state.exclude_count; i++) {
        if (strcmp(g_state.exclude_paths[i], path) == 0)
            return 1;
    }
    return 0;
}
